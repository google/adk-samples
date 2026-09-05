#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Classify recipe image build failures and report the ones worth reporting.

Two subcommands, used at two points in .github/workflows/recipe-images.yml:

    classify   one per matrix leg: read that image's build log and decide
               whether the recipe broke or the infrastructure did, writing
               a small result.json the report job collects.

    report     once per run: collate every result, comment on the merged
               pull request when a recipe genuinely broke, and open a
               tracking issue only when the same image fails again.

BLAME THE RECIPE ONLY ON EVIDENCE
---------------------------------
Taken verbatim from recipe-canary.yml, which states the reason plainly: the
cost of the two mistakes is not symmetric. A missed real failure is caught by
the next run. One failure misfiled against an author for a registry timeout
teaches everyone that this channel cries wolf, and from then on the real ones
are ignored too.

So every path that reaches a verdict without a log to read classifies as
`infra`, and anything matching a known infrastructure signature does too.
Only a build that failed with output that looks like the Dockerfile's own
fault is called `fail`.

WHY A PULL REQUEST COMMENT FIRST, AND AN ISSUE ONLY ON A REPEAT
---------------------------------------------------------------
For a failure on main the actionable human is whoever merged the change, and
a comment on their pull request reaches them where the context already is. It
needs no ownership metadata to be correct, which matters here:
canary_issues.py records that only 3 of 12 recipes declare an
`ownership.poc` that can even be assigned, because GitHub silently drops an
unknown assignee.

An issue is for the failure nobody acted on. Opening one for every failure
would produce a tracker full of items already fixed by the next commit; a
second consecutive failure of the same image is evidence the comment was
missed, and that is worth something more durable.

"Repeated" is read from the workflow's own run history rather than from a
file or a label, so there is no state to keep in sync and no first run that
behaves differently because its state does not exist yet.

Usage:
    image_build_report.py classify --image X --outcome failure \\
        --log build.log --out result.json
    image_build_report.py report --results DIR --run-url URL [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# The default is for reading and for tests. Writing with it is refused —
# see _require_explicit_repo(). Without that guard, running the report
# subcommand anywhere GITHUB_REPOSITORY is unset would post comments and
# open issues on google/adk-samples, which is the wrong repository for
# everyone who is not CI.
DEFAULT_REPO = "google/adk-samples"
REPO = os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO)
WORKFLOW_FILE = "recipe-images.yml"

# Who is told when a failure needs a human and no better address exists.
# The repository's CODEOWNERS catch-all, and the same value canary_issues.py
# uses for the same purpose.
MAINTAINER = "happyhuman"

TITLE_PREFIX = "Recipe image build failing:"
LABEL_TRACKING = "recipe-image-build"

# How many log lines to quote back. Enough to show the failing instruction
# and its error, short enough that a comment stays readable.
LOG_TAIL_LINES = 25

# How much of the one-line summary to keep. Long enough for a docker error
# naming the failing instruction, short enough that a result.json stays small
# and a printed summary line stays on one row.
MAX_DETAIL_CHARS = 300

# Upper bound on issues one run will open. Sized well above routine breakage
# — three images are declared today — and below "something is systemically
# wrong". Mirrors DEFAULT_MAX_ISSUES in canary_issues.py, for the same
# reason: automation that writes to the tracker should refuse a batch whose
# size implies its own inputs are broken.
MAX_ISSUES_PER_RUN = 5

# How many past runs to walk when asking "did this image fail last time it
# was built". Deep enough to see through a stretch of merges that touched
# other recipes, shallow enough to cost a handful of API calls. Beyond this
# the answer is "no recent evidence", which biases toward not opening an
# issue — the recoverable direction.
RUN_HISTORY_DEPTH = 10

# The build job's name is `build <image>`; the matrix leg names come from
# the workflow's `name:` expression. Parsing them back is the only way to
# map a historical job to an image, so the two must agree — enforced by
# test_the_build_job_name_matches_what_the_reporter_parses.
BUILD_JOB_PREFIX = "build "

# Log signatures that mean the infrastructure failed, not the recipe.
#
# Each is a real docker/registry failure mode with a distinctive string. The
# list is deliberately generous: a false `infra` costs a missed report that
# the next run catches, while a false `fail` costs the channel's credibility.
#
# Matched case-insensitively against the build log.
# One reason string shared by the three spellings below, so the wording
# they report cannot drift apart.
_REGISTRY_5XX = "registry returned a server error"

INFRA_SIGNATURES: tuple[tuple[str, str], ...] = (
    # Registry and network.
    #
    # Three spellings of "the registry returned an error status". The status
    # code and the host appear in either order depending on which layer of
    # docker reports it, and a signature matching only one order would blame
    # the recipe for the other — which is how the first version of this list
    # misclassified a real 503 from docker.io.
    (r"unexpected (?:http )?status:?\s*(?:50\d|429)", _REGISTRY_5XX),
    (
        r"\b(?:50\d|429)\b[^\n]{0,160}"
        r"(?:registry|docker\.io|pkg\.dev|ghcr\.io)",
        _REGISTRY_5XX,
    ),
    (
        r"(?:registry|docker\.io|pkg\.dev|ghcr\.io)[^\n]{0,160}"
        r"\b(?:50\d|429)\b",
        _REGISTRY_5XX,
    ),
    (r"toomanyrequests|rate limit|too many requests", "registry rate limit"),
    (
        r"connection reset by peer|connection refused|broken pipe",
        "network dropped",
    ),
    (
        r"i/o timeout|timeout awaiting|context deadline exceeded",
        "network timeout",
    ),
    (r"temporary failure in name resolution|no such host|dns", "DNS failure"),
    (
        r"tls handshake|x509|certificate (?:has expired|verify failed)",
        "TLS failure",
    ),
    (
        r"unexpected eof|unexpected status from (?:head|get) request",
        "truncated registry response",
    ),
    # Pulling the base image.
    (r"failed to resolve source metadata", "base image could not be resolved"),
    (
        r"pull access denied|manifest unknown|manifest for .* not found",
        "base image unavailable",
    ),
    (r"error pulling image configuration", "base image layer download failed"),
    # The runner itself.
    (r"no space left on device", "the runner ran out of disk"),
    (
        r"cannot allocate memory|out of memory|oom-kill",
        "the runner ran out of memory",
    ),
    (
        r"failed to (?:start|connect to) (?:the )?docker daemon|cannot connect to the docker daemon",
        "the docker daemon was unavailable",
    ),
    # Authentication, which is configuration rather than the recipe.
    (
        r"unauthorized: |denied: permission|token.{0,20}expired|invalid_grant",
        "registry authentication failed",
    ),
    (
        r"workload identity|failed to generate google-github-actions",
        "cloud authentication failed",
    ),
)

_COMPILED = tuple(
    (re.compile(pattern, re.IGNORECASE), reason)
    for pattern, reason in INFRA_SIGNATURES
)

PASS = "pass"
FAIL = "fail"
INFRA = "infra"


class ReportError(RuntimeError):
    """The report cannot be produced. Always fatal: a reporter that cannot
    tell anyone must fail its run rather than report success in silence."""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(outcome: str, log: str | None) -> tuple[str, str]:
    """Decide what a single image's build result means.

    Returns (verdict, detail). `outcome` is the GitHub step outcome for the
    docker build — "success", "failure", "skipped" or "" if it never ran.

    The order of these branches is the evidence rule made executable: every
    way of reaching a verdict WITHOUT a log to read lands on `infra`, and the
    only path to `fail` requires log text that matched no infrastructure
    signature.
    """
    if outcome == "success":
        return PASS, ""

    if outcome in ("", "skipped", "cancelled"):
        # The build never ran, so nothing was learned about the recipe.
        # Checkout, the mode step, or the runner died ahead of it.
        return (
            INFRA,
            f"the build step did not run (outcome: {outcome or 'unset'})",
        )

    if not log or not log.strip():
        # docker writes to this file before it starts, so an empty log means
        # it never got far enough to say anything. No evidence, no blame.
        return INFRA, "the build failed without producing any output"

    for pattern, reason in _COMPILED:
        match = pattern.search(log)
        if match:
            return INFRA, reason

    return FAIL, _failure_detail(log)


def _failure_detail(log: str) -> str:
    """The most useful line or two from a failing build log.

    Prefers the line naming the instruction that failed, since that is what
    tells the reader where to look; falls back to the tail.
    """
    lines = [line.rstrip() for line in log.splitlines() if line.strip()]
    for line in reversed(lines):
        if re.search(
            r"^ERROR|did not complete successfully|executor failed",
            line,
            re.IGNORECASE,
        ):
            return line[:MAX_DETAIL_CHARS]
    return lines[-1][:MAX_DETAIL_CHARS] if lines else ""


def log_tail(log: str, limit: int = LOG_TAIL_LINES) -> str:
    lines = [line.rstrip() for line in log.splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


# ---------------------------------------------------------------------------
# gh plumbing
# ---------------------------------------------------------------------------


def gh(*args: str, check: bool = True) -> str:
    """Run `gh`. Mirrors canary_issues.gh, including never swallowing a
    failure: a reporter that cannot reach GitHub must fail loudly."""
    result = subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise ReportError(
            f"gh {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def pull_request_for_commit(sha: str) -> int | None:
    """The pull request a commit on main came from, or None.

    A squash merge leaves the number in the commit subject, but parsing that
    breaks on a subject that merely mentions one. The API answers directly.
    None is a legitimate answer — a direct push to main has no pull request —
    and the caller falls back to an issue rather than staying silent.
    """
    raw = gh(
        "api",
        f"repos/{REPO}/commits/{sha}/pulls",
        "-H",
        "Accept: application/vnd.github+json",
        check=False,
    )
    try:
        prs = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return None
    for pr in prs:
        if isinstance(pr, dict) and pr.get("number"):
            return int(pr["number"])
    return None


def images_that_failed_in_the_previous_run(
    current_run_id: str, wanted: set[str] | None = None
) -> set[str]:
    """Images that failed THE LAST TIME EACH WAS BUILT, not merely last run.

    The distinction is the whole correctness of this function, because builds
    are affected-only. The realistic sequence is:

        run A   demo fails
        run B   a merge touching another recipe; demo is not built at all
        run C   a merge touching another recipe; demo is not built at all
        run D   demo fails again

    Asking "did demo fail in the run immediately before D" answers no — run C
    never built it — so D reads as a first failure, and it reads that way
    every time. An earlier version did exactly that, which made the issue
    path unreachable in practice rather than merely rare.

    So this walks back through recent runs and, for each image, takes the
    verdict from the most recent run that ACTUALLY BUILT it. Runs that
    skipped an image are transparent to that image and opaque to no other.

    The run history IS the state. The alternative — a label, a file, a
    counter in an issue body — has to be kept in sync with reality, and the
    first run after it is introduced behaves differently from every later one
    because the state does not exist yet.

    Returns an empty set on any difficulty reaching the API: that biases
    toward commenting without opening an issue, which is the recoverable
    direction, since the next failure opens it.
    """
    runs = _recent_completed_runs()
    if not runs:
        return set()

    # image -> conclusion from the most recent run that built it.
    seen: dict[str, str] = {}
    for run in runs:
        if str(run.get("id")) == str(current_run_id):
            continue
        for image, conclusion in _build_job_outcomes(run.get("id")).items():
            # First writer wins: runs arrive newest-first, so the first time
            # an image appears is the most recent build of it.
            seen.setdefault(image, conclusion)
        # Stop once every image the caller asked about has a verdict. Each
        # extra run costs an API call, and in the common case — the previous
        # run built the same image — the answer is complete after one.
        # Without this the walk always cost RUN_HISTORY_DEPTH calls, which is
        # the n-calls-for-one-question shape canary_issues warns about.
        if wanted and wanted <= set(seen):
            break

    return {img for img, concl in seen.items() if concl == "failure"}


def _recent_completed_runs() -> list[dict]:
    """Recent completed push-to-main runs of this workflow, newest first."""
    try:
        raw = gh(
            "api",
            f"repos/{REPO}/actions/workflows/{WORKFLOW_FILE}/runs"
            f"?branch=main&event=push&status=completed"
            f"&per_page={RUN_HISTORY_DEPTH}",
            check=False,
        )
        parsed = json.loads(raw or "{}")
        # `.get` on a list is an AttributeError, and the API returns a bare
        # list on some error shapes. Guarding the TYPE rather than only the
        # decode is what keeps a bad response from taking the whole report
        # down after the pull request comment has already been posted.
        if not isinstance(parsed, dict):
            return []
        runs = parsed.get("workflow_runs") or []
        return [r for r in runs if isinstance(r, dict)]
    except (json.JSONDecodeError, ReportError):
        return []


def _build_job_outcomes(run_id: Any) -> dict[str, str]:
    """image -> job conclusion for the build legs of one run."""
    if run_id is None:
        return {}
    try:
        raw = gh(
            "api",
            f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100",
            check=False,
        )
        parsed = json.loads(raw or "{}")
        if not isinstance(parsed, dict):
            return {}
        jobs = parsed.get("jobs") or []
    except (json.JSONDecodeError, ReportError):
        return {}

    outcomes: dict[str, str] = {}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        name = str(job.get("name") or "")
        if not name.startswith(BUILD_JOB_PREFIX):
            continue
        image = name[len(BUILD_JOB_PREFIX) :].strip()
        if image:
            outcomes[image] = str(job.get("conclusion") or "")
    return outcomes


# ---------------------------------------------------------------------------
# Report bodies
# ---------------------------------------------------------------------------


def comment_body(failures: list[dict], run_url: str, sha: str) -> str:
    plural = "image" if len(failures) == 1 else "images"
    merge = f"The merge of {sha[:8]}" if sha else "This merge"
    lines = [
        f"### Recipe {plural} failed to build on `main`",
        "",
        f"{merge} broke the container build for {len(failures)} declared "
        f"{plural}. Nothing was published — the build and the publish step "
        f"are separate, and publishing did not run.",
        "",
    ]
    for entry in failures:
        # The reproduce command goes with EACH image, not once at the end.
        # An earlier version printed only the first failure's command below
        # the whole list, which reads as though it covers all of them — so
        # someone with three broken images would run one build, see it pass
        # after a fix, and believe they were done.
        lines += [
            f"**`{entry['image']}`** — `{entry.get('dockerfile', '?')}`",
            "",
            "```",
            entry.get("tail") or entry.get("detail") or "(no output captured)",
            "```",
            "",
            "```bash",
            f"docker build -f {entry.get('dockerfile', 'Dockerfile')} "
            f"{entry.get('context', '.')}",
            "```",
            "",
        ]
    lines += [f"[Full logs]({run_url})"]
    return "\n".join(lines)


def issue_title(image: str) -> str:
    """Stable across runs — it is the dedupe key, so nothing varying (a SHA,
    a date, the error text) may appear in it."""
    return f"{TITLE_PREFIX} {image}"


def issue_body(entry: dict, run_url: str, previous_note: str) -> str:
    return "\n".join(
        [
            f"`{entry['image']}` has now failed to build on `main` "
            f"{previous_note}.",
            "",
            f"- Dockerfile: `{entry.get('dockerfile', '?')}`",
            f"- Build context: `{entry.get('context', '?')}`",
            "- Declared in: `.github/policy.yml` "
            "(`deployability.publish.images`)",
            "",
            "```",
            entry.get("tail") or entry.get("detail") or "(no output captured)",
            "```",
            "",
            f"[Latest run]({run_url})",
            "",
            "Until this builds, the image is not published. Either fix the "
            "Dockerfile or remove the entry from `deployability.publish` — "
            "an image declared and permanently broken is worse than one not "
            "declared, because the failure is reported every time anything "
            "in the recipe changes.",
            "",
            f"cc @{MAINTAINER}",
        ]
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_classify(args: argparse.Namespace) -> int:
    log = ""
    if args.log:
        try:
            log = Path(args.log).read_text(encoding="utf-8", errors="replace")
        except OSError:
            # Treated as no evidence rather than as an error: the classifier's
            # whole job is to produce a verdict, and a missing log has one.
            log = ""

    verdict, detail = classify(args.outcome, log)
    result = {
        "image": args.image,
        "dockerfile": args.dockerfile or "",
        "context": args.context or "",
        "outcome": verdict,
        "detail": detail,
        "tail": log_tail(log) if verdict == FAIL else "",
    }
    Path(args.out).write_text(json.dumps(result), encoding="utf-8")
    print(f"{args.image}: {verdict}" + (f" — {detail}" if detail else ""))
    return 0


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Every result.json under `results_dir`, skipping anything unusable.

    Shape is checked, not assumed. A file that is valid JSON but not an
    object — or an object with no `image` — would otherwise raise deep in the
    reporting logic, potentially after a comment has already been posted, and
    the run would look like a reporter bug rather than a corrupt artifact.
    """
    entries = []
    for path in sorted(results_dir.rglob("result.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: cannot read {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(entry, dict) or not entry.get("image"):
            print(
                f"WARNING: {path} is not a usable result (no image); "
                f"ignoring it",
                file=sys.stderr,
            )
            continue
        entries.append(entry)
    return entries


def _require_explicit_repo(dry_run: bool) -> None:
    """Refuse to write to the fallback repository.

    `REPO` falls back to DEFAULT_REPO so the module imports cleanly and the
    tests need no environment. That is harmless for reading and for
    `--dry-run`, and dangerous for anything else: run the report subcommand
    on a laptop, or in a fork whose workflow forgot the variable, and it
    would post comments and open issues on google/adk-samples.

    CI always sets GITHUB_REPOSITORY, so this costs the real caller nothing.
    """
    if dry_run:
        return
    if not os.environ.get("GITHUB_REPOSITORY"):
        raise ReportError(
            "GITHUB_REPOSITORY is not set, so the target repository would "
            f"fall back to {DEFAULT_REPO}. Refusing to comment or open "
            "issues there. Set it, or pass --dry-run."
        )


def cmd_report(args: argparse.Namespace) -> int:
    _require_explicit_repo(args.dry_run)
    entries = load_results(Path(args.results))
    if not entries:
        # Distinguished from "everything passed": the build job writes a
        # result for every leg including the ones that passed, so no results
        # at all means the collection broke, not that nothing ran.
        print(
            "error: no results found. The build jobs did not upload any, "
            "so this run cannot say whether anything failed.",
            file=sys.stderr,
        )
        return 1

    failures = [e for e in entries if e.get("outcome") == FAIL]
    infra = [e for e in entries if e.get("outcome") == INFRA]
    passed = [e for e in entries if e.get("outcome") == PASS]

    print(
        f"{len(entries)} image(s): {len(passed)} pass, "
        f"{len(failures)} fail, {len(infra)} infra"
    )
    for entry in entries:
        print(
            f"  {entry.get('outcome', '?'):5}  {entry.get('image')}"
            + (f" — {entry['detail']}" if entry.get("detail") else "")
        )

    if infra and not failures:
        # Deliberately silent. Reporting infrastructure noise to an author is
        # precisely what makes a channel ignorable.
        print(
            "\nOnly infrastructure failures; nothing reported to anyone. "
            "Re-run the workflow."
        )

    # Recovery is handled BEFORE the early return below, because "no
    # failures" is exactly the case a recovery looks like. An earlier version
    # returned first and left every tracking issue open forever — the close
    # path was unreachable code that read as if it worked.
    # ONE snapshot for the whole run, taken here and passed down.
    # canary_issues.open_issues_by_title documents why a lookup inside a
    # loop is wrong: n subprocesses and n chances to hit a rate limit, to
    # answer a question one call already answers. Taking it twice — once
    # for recoveries and once for failures — was a smaller version of the
    # same mistake.
    open_issues = _open_tracking_issues()

    _close_recovered(passed, args, open_issues)

    if not failures:
        return 0

    # Above the cap, comment and open NOTHING. The message used to say
    # "commenting only" while the loop below went on opening issues up to the
    # cap anyway — the log said one thing and the tracker showed another.
    # Refusing outright is also the better behaviour: this many failures at
    # once is far more likely to be one systemic cause than N independent
    # breakages, and N issues would each be wrong about what to fix.
    file_issues = len(failures) <= MAX_ISSUES_PER_RUN
    if not file_issues:
        print(
            f"WARNING: {len(failures)} images failed at once, above the "
            f"{MAX_ISSUES_PER_RUN} this run will file for. That is more "
            f"likely one systemic cause than {len(failures)} independent "
            f"breakages, so no issues are being opened — the pull request "
            f"comment is the whole report for this run.",
            file=sys.stderr,
        )

    # --- the pull request comment ------------------------------------------
    pr = pull_request_for_commit(args.sha) if args.sha else None
    body = comment_body(failures, args.run_url, args.sha or "")
    if pr is None:
        print("\nNo pull request found for this commit (a direct push?).")
    elif args.dry_run:
        print(f"\n[dry-run] would comment on #{pr}:\n{body}\n")
    else:
        gh("issue", "comment", str(pr), "--repo", REPO, "--body", body)
        print(f"\nCommented on #{pr}.")

    # --- the issue, only on a repeat ---------------------------------------
    if not file_issues:
        return 0

    repeats = images_that_failed_in_the_previous_run(
        args.run_id or "", {e["image"] for e in failures}
    )
    # One snapshot for the whole run, not one lookup per image.
    # canary_issues.open_issues_by_title documents why: a lookup inside the
    # loop is n subprocesses and n chances to hit a rate limit, to answer a
    # question one call already answers.
    for entry in failures:
        image = entry["image"]
        if image not in repeats:
            print(f"{image}: first failure — commented, no issue opened.")
            continue

        title = issue_title(image)
        existing = open_issues.get(title)
        note = "again since it was last built"
        if existing:
            if args.dry_run:
                print(f"[dry-run] would comment on issue #{existing}")
            else:
                gh(
                    "issue",
                    "comment",
                    str(existing),
                    "--repo",
                    REPO,
                    "--body",
                    f"Still failing as of {args.sha[:8] if args.sha else '?'}"
                    f" — [run]({args.run_url}).",
                )
                print(f"{image}: commented on existing issue #{existing}.")
            continue

        if args.dry_run:
            print(f"[dry-run] would open an issue titled {title!r}")
        else:
            _ensure_label()
            gh(
                "issue",
                "create",
                "--repo",
                REPO,
                "--title",
                title,
                "--label",
                LABEL_TRACKING,
                "--body",
                issue_body(entry, args.run_url, note),
            )
            print(f"{image}: opened a tracking issue.")

    return 0


def _close_recovered(
    passed: list[dict],
    args: argparse.Namespace,
    open_issues: dict[str, int],
) -> None:
    """Close the tracking issue of any image that is building again."""
    for entry in passed:
        existing = open_issues.get(issue_title(entry["image"]))
        if not existing:
            continue
        if args.dry_run:
            print(f"[dry-run] would close issue #{existing}")
            continue
        gh(
            "issue",
            "close",
            str(existing),
            "--repo",
            REPO,
            "--comment",
            f"Building again as of {args.sha[:8] if args.sha else '?'}.",
        )
        print(f"{entry['image']}: closed issue #{existing}.")


def _open_tracking_issues() -> dict[str, int]:
    """Open tracking issues indexed by EXACT title. One `gh` call per run.

    Exact keys, never a prefix scan: `demo` must not be handed the issue
    belonging to `demo-sandbox-runtime`, which would report one image's
    failure on another image's thread.
    """
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
        "100",
        "--json",
        "number,title",
        check=False,
    )
    try:
        parsed = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, list):
        return {}
    return {
        issue["title"]: int(issue["number"])
        for issue in parsed
        if isinstance(issue, dict)
        and issue.get("title")
        and issue.get("number")
    }


def _ensure_label() -> None:
    gh(
        "label",
        "create",
        LABEL_TRACKING,
        "--repo",
        REPO,
        "--description",
        "A declared recipe image is failing to build",
        "--color",
        "B60205",
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("classify", help="Classify one image's build result.")
    c.add_argument("--image", required=True)
    c.add_argument("--outcome", required=True, help="The build step outcome.")
    c.add_argument("--log", help="Path to the captured build log.")
    c.add_argument("--dockerfile", default="")
    c.add_argument("--context", default="")
    c.add_argument("--out", required=True, help="Where to write result.json.")
    c.set_defaults(func=cmd_classify)

    r = sub.add_parser("report", help="Collate results and report failures.")
    r.add_argument("--results", required=True, help="Directory of results.")
    r.add_argument("--run-url", default="")
    r.add_argument("--run-id", default="")
    r.add_argument("--sha", default="")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
