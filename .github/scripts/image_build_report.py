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

REPO = os.environ.get("GITHUB_REPOSITORY", "google/adk-samples")
WORKFLOW_FILE = "recipe-images.yml"

# Who is told when a failure needs a human and no better address exists.
MAINTAINER = "happyhuman"

TITLE_PREFIX = "Recipe image build failing:"
LABEL_TRACKING = "recipe-image-build"

# How many log lines to quote back. Enough to show the failing instruction
# and its error, short enough that a comment stays readable.
LOG_TAIL_LINES = 25

# Upper bound on issues one run will open. Sized well above routine breakage
# — three images are declared today — and below "something is systemically
# wrong". Mirrors DEFAULT_MAX_ISSUES in canary_issues.py, for the same
# reason: automation that writes to the tracker should refuse a batch whose
# size implies its own inputs are broken.
MAX_ISSUES_PER_RUN = 5

# Log signatures that mean the infrastructure failed, not the recipe.
#
# Each is a real docker/registry failure mode with a distinctive string. The
# list is deliberately generous: a false `infra` costs a missed report that
# the next run catches, while a false `fail` costs the channel's credibility.
#
# Matched case-insensitively against the build log.
INFRA_SIGNATURES: tuple[tuple[str, str], ...] = (
    # Registry and network.
    #
    # Three spellings of "the registry returned an error status". The status
    # code and the host appear in either order depending on which layer of
    # docker reports it, and a signature matching only one order would blame
    # the recipe for the other — which is how the first version of this list
    # misclassified a real 503 from docker.io.
    (
        r"unexpected (?:http )?status:?\s*(?:50\d|429)",
        "registry returned a server error",
    ),
    (
        r"\b(?:50\d|429)\b[^\n]{0,160}"
        r"(?:registry|docker\.io|pkg\.dev|ghcr\.io)",
        "registry returned a server error",
    ),
    (
        r"(?:registry|docker\.io|pkg\.dev|ghcr\.io)[^\n]{0,160}"
        r"\b(?:50\d|429)\b",
        "registry returned a server error",
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
            return line[:300]
    return lines[-1][:300] if lines else ""


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


def images_that_failed_in_the_previous_run(current_run_id: str) -> set[str]:
    """Images whose build job failed in the last completed run before this one.

    The run history IS the state. The alternative — a label, a file, a issue
    body counter — has to be kept in sync with reality, and the first run
    after it is introduced behaves differently from every later one because
    the state does not exist yet.

    Returns an empty set on any difficulty reaching the API. That biases
    toward commenting without opening an issue, which is the recoverable
    direction: the next failure opens it.
    """
    try:
        raw = gh(
            "api",
            f"repos/{REPO}/actions/workflows/{WORKFLOW_FILE}/runs"
            "?branch=main&event=push&status=completed&per_page=5",
            check=False,
        )
        parsed = json.loads(raw or "{}")
        # `.get` on a list is an AttributeError, and the API returns a bare
        # list on some error shapes. Guarding the TYPE rather than only the
        # decode is what keeps a bad response from taking the whole report
        # down after the pull request comment has already been posted.
        runs = (
            parsed.get("workflow_runs") or []
            if isinstance(parsed, dict)
            else []
        )
    except (json.JSONDecodeError, ReportError):
        return set()

    previous = next(
        (r for r in runs if str(r.get("id")) != str(current_run_id)), None
    )
    if not previous:
        return set()

    try:
        raw = gh(
            "api",
            f"repos/{REPO}/actions/runs/{previous['id']}/jobs?per_page=100",
            check=False,
        )
        parsed = json.loads(raw or "{}")
        jobs = parsed.get("jobs") or [] if isinstance(parsed, dict) else []
    except (json.JSONDecodeError, ReportError):
        return set()

    failed = set()
    for job in jobs:
        name = str(job.get("name") or "")
        if job.get("conclusion") == "failure" and name.startswith("build "):
            failed.add(name[len("build ") :].strip())
    return failed


# ---------------------------------------------------------------------------
# Report bodies
# ---------------------------------------------------------------------------


def comment_body(failures: list[dict], run_url: str, sha: str) -> str:
    plural = "image" if len(failures) == 1 else "images"
    lines = [
        f"### Recipe {plural} failed to build on `main`",
        "",
        f"The merge of {sha[:8]} broke the container build for "
        f"{len(failures)} declared {plural}. Nothing was published.",
        "",
    ]
    for entry in failures:
        lines += [
            f"**`{entry['image']}`** — `{entry.get('dockerfile', '?')}`",
            "",
            "```",
            entry.get("tail") or entry.get("detail") or "(no output captured)",
            "```",
            "",
        ]
    lines += [
        f"[Full logs]({run_url})",
        "",
        "This is the build only — publishing is separate and did not run. "
        "Reproduce locally with:",
        "",
        "```bash",
        f"docker build -f {failures[0].get('dockerfile', 'Dockerfile')} "
        f"{failures[0].get('context', '.')}",
        "```",
    ]
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
    entries = []
    for path in sorted(results_dir.rglob("result.json")):
        try:
            entries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: cannot read {path}: {exc}", file=sys.stderr)
    return entries


def cmd_report(args: argparse.Namespace) -> int:
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
    _close_recovered(passed, args)

    if not failures:
        return 0

    if len(failures) > MAX_ISSUES_PER_RUN:
        print(
            f"WARNING: {len(failures)} images failed at once, above the "
            f"{MAX_ISSUES_PER_RUN} this run will open issues for. Something "
            f"systemic is more likely than {len(failures)} independent "
            f"breakages; commenting only.",
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
    repeats = images_that_failed_in_the_previous_run(args.run_id or "")
    opened = 0
    for entry in failures:
        image = entry["image"]
        if image not in repeats:
            print(f"{image}: first failure — commented, no issue opened.")
            continue
        if opened >= MAX_ISSUES_PER_RUN:
            print(f"{image}: issue cap reached, skipping.")
            continue

        title = issue_title(image)
        existing = _find_open_issue(title)
        note = "twice in a row"
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
        opened += 1

    return 0


def _close_recovered(passed: list[dict], args: argparse.Namespace) -> None:
    """Close the tracking issue of any image that is building again."""
    for entry in passed:
        existing = _find_open_issue(issue_title(entry["image"]))
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


def _find_open_issue(title: str) -> int | None:
    """Exact title match among open tracking issues.

    Never a prefix match: `demo` must not be handed the issue belonging to
    `demo-sandbox`, which would report one image's failure on another's
    thread.
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
        for issue in json.loads(raw or "[]"):
            if issue.get("title") == title:
                return int(issue["number"])
    except json.JSONDecodeError:
        return None
    return None


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
