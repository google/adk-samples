#!/usr/bin/env python3
"""
Drive the issue lifecycle for .github/workflows/recipe-canary.yml.

Reads the canary's per-job results and decides, for each recipe, what to do
about its tracking issue: open one, nudge it, escalate it, or close it.

The lifecycle, clocked off the ISSUE's created_at
-------------------------------------------------
    day   0   open an issue, notify the recipe's owner
    day  30   reminder comment
    day  60   propose marking the recipe `status: inactive`
    day  90   warn that deletion is scheduled
    day 120   propose deleting the recipe, tagging a maintainer
    passing   close the issue, and propose restoring `status: active`

The issue is the clock deliberately: no new manifest field, no archaeology
over git history to work out when something started failing, and a stage that
cannot fire twice because the label recording it is right there on the issue.

Why "propose" and not "do"
--------------------------
Three stages want to change files on `main`, which is protected: 1 approving
review AND a code-owner review. A workflow authenticating with GITHUB_TOKEN
cannot push there, and its approvals do not count toward either requirement —
that is exactly why 98 Dependabot PRs sat unmergeable for months. So this
script never writes to the repo directly. It opens a pull request, or (when
no elevated token is configured) leaves a comment saying what a human needs
to do. Degrading to a comment matters: it means the canary is useful the day
it merges, before any App identity exists.

Notifying the owner
-------------------
`manifest.ownership.poc` is the source of truth, but only 3 of 12 declared
owners can actually be ASSIGNED — the rest have no repo access, and GitHub
silently drops an invalid assignee, creating the issue unassigned with no
error. So the @-mention in the body is the real delivery mechanism and
assignment is a bonus applied only when it will work.

Usage:
    python canary_issues.py --results results.json
    python canary_issues.py --results results.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO = os.environ.get("GITHUB_REPOSITORY", "google/adk-samples")

# Who hears about it when a recipe has no usable owner, and who is tagged for
# a decision once the owner has not responded.
MAINTAINER = "happyhuman"

TITLE_PREFIX = "Recipe canary:"

LABEL_TRACKING = "recipe-canary"
LABEL_ROTTING = "recipe:rotting"
LABEL_DELETION = "recipe:deletion-scheduled"

# Stage -> (age in days, label that records it having fired).
#
# The label IS the state. Without it the day-30 comment would repeat on every
# run past day 30, which is the monthly nagging this whole system exists to
# stop.
STAGE_REMIND = (30, "recipe:reminded")
STAGE_INACTIVE = (60, LABEL_ROTTING)
STAGE_WARN_DELETE = (90, LABEL_DELETION)
STAGE_DELETE = (120, "recipe:deletion-proposed")


class GhError(RuntimeError):
    """A `gh` invocation failed. Never silently swallowed: a canary that
    cannot file issues must fail its run, not report success having told
    nobody."""


def gh(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise GhError(
            f"gh {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def issue_title(recipe: str) -> str:
    """Stable across runs — it is the dedupe key.

    Anything varying (a version, a date, the error text) would stop the canary
    recognising its own issue and it would file a fresh one every month.
    """
    return f"{TITLE_PREFIX} {recipe} is failing"


def read_owner(recipe: str, repo_root: Path | None = None) -> str | None:
    """`ownership.poc` from the recipe's manifest, without a YAML dependency.

    Returns None when the manifest is missing, unreadable, or has no poc — the
    caller falls back to the maintainer rather than staying silent, because a
    broken recipe with a broken manifest is the last thing that should vanish.
    """
    root = REPO_ROOT if repo_root is None else repo_root
    manifest = root / recipe / "manifest.yaml"
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    in_ownership = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ownership:"):
            in_ownership = True
            continue
        if in_ownership:
            if stripped and not line.startswith((" ", "\t")):
                break  # left the ownership block
            if stripped.startswith("poc:"):
                value = stripped.split(":", 1)[1].strip()
                value = value.split("#", 1)[0].strip().strip("\"'")
                return value or None
    return None


def is_assignable(user: str) -> bool:
    """GitHub drops an unknown assignee silently, so ask before assigning."""
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/assignees/{user}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def find_issue(recipe: str) -> dict | None:
    """The open canary issue for this recipe, if one exists."""
    raw = gh(
        "issue",
        "list",
        "--repo",
        REPO,
        "--state",
        "open",
        "--label",
        LABEL_TRACKING,
        "--limit",
        "200",
        "--json",
        "number,title,createdAt,labels",
    )
    wanted = issue_title(recipe)
    for issue in json.loads(raw or "[]"):
        if issue["title"] == wanted:
            issue["labelNames"] = {lbl["name"] for lbl in issue["labels"]}
            return issue
    return None


def age_days(created_at: str, now: datetime | None = None) -> int:
    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    current = now or datetime.now(timezone.utc)
    return (current - created).days


def failing_recipes(results: list[dict]) -> dict[str, list[dict]]:
    """Recipes with at least one FAILED job, mapped to those jobs.

    `infra` outcomes are ignored on purpose. A PyPI timeout or a runner
    problem is not the recipe's fault, and filing against an owner for it is
    how a notification channel loses its credibility.
    """
    failures: dict[str, list[dict]] = {}
    for entry in results:
        if entry.get("outcome") == "fail":
            failures.setdefault(entry["recipe"], []).append(entry)
    return failures


def _mention(owner: str | None) -> tuple[str, str]:
    """(mention line, note) for the issue body."""
    if owner:
        return (
            f"@{owner}",
            "You are listed as `ownership.poc` for this recipe.",
        )
    return (
        f"@{MAINTAINER}",
        "This recipe's `manifest.yaml` declares no usable `ownership.poc`, "
        "so a maintainer is being notified instead. Setting a valid `poc` "
        "would route future notices to the right person.",
    )


def build_body(recipe: str, jobs: list[dict], run_url: str) -> str:
    """`jobs` is EVERY job for the recipe, passing ones included.

    The passing versions are what make the report actionable: "broken on 3.13,
    fine on 3.11" tells the owner it is an interpreter problem, where "broken"
    alone sends them looking at their own code. An earlier version of this
    function was handed only the failures and silently lost that.
    """
    owner = read_owner(recipe)
    mention, note = _mention(owner)
    failed = [j for j in jobs if j.get("outcome") == "fail"]
    versions = ", ".join(sorted({j["python"] for j in failed}))
    passing = sorted({j["python"] for j in jobs if j.get("outcome") == "pass"})

    lines = [
        f"{mention} — the monthly recipe canary could not get "
        f"`{recipe}` working.",
        "",
        note,
        "",
        "## What failed",
        "",
        f"Python {versions}.",
    ]
    if passing:
        lines += [
            "",
            f"It still passes on Python {', '.join(passing)} — so this is a "
            f"version-specific break, not a wholesale one. The recipe's "
            f"`requires-python` claims support for the failing version.",
        ]
    lines += ["", "| Python | step | detail |", "|---|---|---|"]
    for job in sorted(failed, key=lambda j: j["python"]):
        detail = (job.get("detail") or "").replace("|", "\\|")[:300]
        lines.append(f"| {job['python']} | {job.get('step', '?')} | {detail} |")

    lines += [
        "",
        f"[Full logs]({run_url})",
        "",
        "## What this means",
        "",
        "The canary installs each recipe from its **committed lockfile** "
        "(`uv sync --frozen`) and runs its tests. It does not update "
        "dependencies and never opens version-bump PRs — this repo leaves "
        "dependency freshness to recipe owners. So this is not a stale-"
        "dependency nag: as far as the canary can tell, the recipe does not "
        "work for someone who clones it today.",
        "",
        "## What happens if nobody acts",
        "",
        "| after | then |",
        "|---|---|",
        "| 30 days | a reminder comment here |",
        "| 60 days | a PR marking the recipe `status: inactive` |",
        "| 90 days | notice that deletion is scheduled |",
        f"| 120 days | a PR deleting the recipe, for @{MAINTAINER} to decide |",
        "",
        "Fix the recipe and this issue closes itself on the next run.",
        "",
        "---",
        "<sub>Opened by `.github/workflows/recipe-canary.yml`. Reply here if "
        "the canary is wrong — that is a bug worth fixing.</sub>",
    ]
    return "\n".join(lines)


def ensure_labels() -> None:
    """Create the canary's labels if the repo does not have them yet."""
    for name, colour, description in [
        (LABEL_TRACKING, "0e8a16", "Opened by the monthly recipe canary"),
        (LABEL_ROTTING, "fbca04", "Canary-failing for 60+ days"),
        (LABEL_DELETION, "d93f0b", "Canary-failing for 90+ days"),
        (STAGE_REMIND[1], "c5def5", "Canary reminder sent"),
        (STAGE_DELETE[1], "b60205", "Deletion PR proposed"),
    ]:
        subprocess.run(
            [
                "gh",
                "label",
                "create",
                name,
                "--repo",
                REPO,
                "--color",
                colour,
                "--description",
                description,
                "--force",
            ],
            capture_output=True,
            text=True,
            check=False,
        )


def open_issue(recipe: str, jobs: list[dict], run_url: str, dry: bool) -> None:
    owner = read_owner(recipe)
    body = build_body(recipe, jobs, run_url)
    args = [
        "issue",
        "create",
        "--repo",
        REPO,
        "--title",
        issue_title(recipe),
        "--body",
        body,
        "--label",
        LABEL_TRACKING,
    ]
    if owner and is_assignable(owner):
        args += ["--assignee", owner]
    if dry:
        print(f"  [dry-run] would open: {issue_title(recipe)}")
        return
    print(f"  opened: {gh(*args)}")


def comment(number: int, body: str, dry: bool) -> None:
    if dry:
        print(f"  [dry-run] would comment on #{number}")
        return
    gh("issue", "comment", str(number), "--repo", REPO, "--body", body)


def add_label(number: int, label: str, dry: bool) -> None:
    if dry:
        print(f"  [dry-run] would label #{number} {label}")
        return
    gh("issue", "edit", str(number), "--repo", REPO, "--add-label", label)


def escalate(recipe: str, issue: dict, dry: bool, now=None) -> None:
    """Fire whichever stages the issue's age has reached and not yet recorded.

    Stages are checked oldest-first and each is gated on its own label, so a
    canary that was down for two months catches up in one run instead of
    skipping straight to the last stage.
    """
    number = issue["number"]
    age = age_days(issue["createdAt"], now)
    labels = issue["labelNames"]
    owner = read_owner(recipe)
    mention = f"@{owner}" if owner else f"@{MAINTAINER}"

    days, label = STAGE_REMIND
    if age >= days and label not in labels:
        comment(
            number,
            f"{mention} — still failing {age} days on. Nothing has changed "
            f"on the canary's side; the recipe still does not install and "
            f"pass from its committed lockfile.\n\nAt 60 days the canary "
            f"will open a PR marking this recipe `status: inactive`.",
            dry,
        )
        add_label(number, label, dry)

    days, label = STAGE_INACTIVE
    if age >= days and label not in labels:
        comment(
            number,
            f"{mention} — {age} days without a fix, so this recipe is being "
            f"marked **`status: inactive`**.\n\nThat is not deletion. It "
            f"records that nobody is currently accountable for the recipe "
            f"working, and it is reversible: fix the recipe and the canary "
            f"restores `status: active` on its next run.\n\n"
            f"{_write_note(recipe, 'status: inactive')}",
            dry,
        )
        add_label(number, label, dry)

    days, label = STAGE_WARN_DELETE
    if age >= days and label not in labels:
        comment(
            number,
            f"@{MAINTAINER} {mention} — {age} days. **Deletion is "
            f"scheduled.** At 120 days the canary will open a PR removing "
            f"`{recipe}` from the repository.\n\nA maintainer decides whether "
            f"that PR is merged; the canary only proposes it. To stop the "
            f"clock, fix the recipe or say here that it should be kept.",
            dry,
        )
        add_label(number, label, dry)

    days, label = STAGE_DELETE
    if age >= days and label not in labels:
        comment(
            number,
            f"@{MAINTAINER} — {age} days. Deletion of `{recipe}` is now due."
            f"\n\n{_write_note(recipe, 'deletion')}\n\nNothing is removed "
            f"without a human merging that PR.",
            dry,
        )
        add_label(number, label, dry)


def _write_note(recipe: str, what: str) -> str:
    """What the canary can actually do about a file change, right now.

    Kept in one place because it changes the moment an elevated identity is
    configured, and a stale "a maintainer must do this by hand" instruction
    on an issue is worse than none.
    """
    if os.environ.get("CANARY_APP_TOKEN"):
        return (
            f"A pull request applying `{what}` to `{recipe}` will be opened "
            f"automatically."
        )
    return (
        f"**A maintainer needs to apply `{what}` to `{recipe}` by hand.** The "
        f"canary has no write identity configured (`CANARY_APP_TOKEN` is "
        f"unset), and `GITHUB_TOKEN` cannot push to a branch that requires a "
        f"code-owner review — so it will not pretend to have done it."
    )


def close_recovered(recipe: str, issue: dict, dry: bool) -> None:
    body = (
        f"`{recipe}` installs from its lockfile and passes its tests again, "
        f"so the canary is closing this.\n\n"
    )
    if LABEL_ROTTING in issue["labelNames"]:
        body += (
            f"This recipe was marked `status: inactive` while it was "
            f"failing. {_write_note(recipe, 'status: active')}\n\n"
            f"Until that lands the recipe still reads as inactive, which "
            f"would eventually put it back on the deletion path despite "
            f"working — so it is worth doing promptly."
        )
    else:
        body += "No further action needed."
    if dry:
        print(f"  [dry-run] would close #{issue['number']}")
        return
    comment(issue["number"], body, dry)
    gh("issue", "close", str(issue["number"]), "--repo", REPO)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    results = json.loads(args.results.read_text(encoding="utf-8"))
    if not results:
        print(
            "No canary results. Refusing to act: an empty result set and a "
            "run where every job vanished are indistinguishable, and acting "
            "would close every open issue as recovered.",
            file=sys.stderr,
        )
        return 1

    failures = failing_recipes(results)
    by_recipe: dict[str, list[dict]] = {}
    for entry in results:
        by_recipe.setdefault(entry["recipe"], []).append(entry)
    all_recipes = set(by_recipe)
    print(f"{len(all_recipes)} recipes checked, {len(failures)} failing.")

    if not args.dry_run:
        ensure_labels()

    for recipe in sorted(all_recipes):
        issue = find_issue(recipe)
        if recipe in failures:
            if issue is None:
                print(f"{recipe}: FAILING, opening issue")
                open_issue(
                    recipe, by_recipe[recipe], args.run_url, args.dry_run
                )
            else:
                print(
                    f"{recipe}: FAILING, issue #{issue['number']} already "
                    f"open ({age_days(issue['createdAt'])}d)"
                )
                escalate(recipe, issue, args.dry_run)
        elif issue is not None:
            print(f"{recipe}: recovered, closing #{issue['number']}")
            close_recovered(recipe, issue, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
