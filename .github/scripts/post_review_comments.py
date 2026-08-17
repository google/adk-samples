#!/usr/bin/env python3
"""
Turn the AI PR reviewer's JSON findings into one GitHub review payload.

Used by .github/workflows/_ai-pr-review-core.yml. The reviewer agent used to
post its own comments through the GitHub MCP server, which took three chained
tool calls; the model kept getting that sequence wrong, retried, and re-sent
its whole context each time, so a single review burned ~1.4M input tokens
across ~80 round trips and then hit the wall clock. The agent now returns
findings as JSON and this script does the posting, which makes the reviewer a
single request/response.

Moving the posting here buys line validation as well. GitHub rejects an ENTIRE
review if one comment names a line outside the diff, and a model picking a
plausible-but-wrong line is routine. Anchors are recomputed from the same diff
the reviewer was shown, so a bad position is dropped on its own instead of
taking every other comment down with it.

Usage:
  python3 post_review_comments.py \
    --result agy_result.json \
    --diff pr_diff_used.txt \
    --label Correctness \
    --out review_payload.json

--out is written ONLY when there is at least one postable comment, so the
caller can treat "file absent" as "nothing to post".

Every failure here is a CI fault, never the contributor's: the reviewer agent
returned something unusable, or a file this workflow wrote is unreadable. None
of it is caused by, or fixable from, the pull request under review — so they go
through ci_message.infra_fault, which annotates this checker rather than the
contributor's code.

Exit codes:
  0  payload written, or nothing worth posting
  2  CI fault — the reviewer's output could not be read as findings
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ci_message import (
    EXIT_OK,
    guard,
    infra_fault,
    report_infra_fault,
)

CHECKER = "post_review_comments.py"

# Group 2 is the old-side length, groups 3/4 the new-side start and length.
# A length is absent for a one-line side ("@@ -1 +1 @@"), which means 1.
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# Non-greedy, but the trailing fence forces the match to run to the LAST
# "]" that closes a block, so a "]" inside a comment body does not truncate
# the array.
FENCED_ARRAY = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)


class ReviewerOutputError(Exception):
    """The reviewer returned something this script cannot read as findings."""


def _strip_diff_prefix(path: str) -> str:
    """Drop git's `a/`/`b/` diff prefix from a path."""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _resolve_path(reported: str, anchors: dict[str, set[int]]) -> str:
    """Match the model's path against the paths the diff actually names.

    The prompt asks for the path without git's `b/` prefix and models
    sometimes leave it on. Stripping unconditionally is not safe either: a
    repository with a top-level `b/` directory has real paths beginning that
    way, and stripping would turn `b/pkg/x.py` into `pkg/x.py` and drop every
    finding in it. Prefer the path as given, fall back to the stripped form.
    """
    stripped = _strip_diff_prefix(reported)
    if reported not in anchors and stripped in anchors:
        return stripped
    return reported


def added_line_anchors(diff: str) -> dict[str, set[int]]:
    """Map each path to the new-file line numbers this diff adds.

    Only `+` lines are valid anchors on the RIGHT side of a review, so this
    doubles as enforcement of the prompt's "added lines only" rule.

    Hunk lengths are tracked rather than sniffing line prefixes, because the
    two are not distinguishable by prefix alone. An added line whose text
    begins with "++ " produces the row "+++ ...", identical in shape to a
    file header; prefix-sniffing mistook it for one, set the path to the
    line's own text, and silently lost every remaining anchor in that file.
    Inside a hunk the line counts say exactly how many rows belong to it, so
    a header can only be recognised when we are between hunks.

    A hunk header that does not parse leaves us outside any hunk, so its
    lines contribute no anchors at all. That is deliberate: guessing a start
    line would anchor comments onto real but WRONG lines, which is worse than
    dropping them, because a wrong anchor still passes validation and gets
    posted.
    """
    anchors: dict[str, set[int]] = {}
    path: str | None = None
    new_line = 0
    old_remaining = 0
    new_remaining = 0

    for row in diff.splitlines():
        if old_remaining <= 0 and new_remaining <= 0:
            # Between hunks: the only place a row can be a file header.
            if row.startswith("+++ "):
                target = row[4:].strip()
                path = (
                    None
                    if target == "/dev/null"
                    else _strip_diff_prefix(target)
                )
                continue
            header = HUNK_HEADER.match(row)
            if header:
                old_remaining = int(header.group(2) or 1)
                new_line = int(header.group(3))
                new_remaining = int(header.group(4) or 1)
            # Everything else between hunks ("diff --git", "index", "--- a/x",
            # "Binary files ... differ") carries no line numbering.
            continue

        # Inside a hunk. "\ No newline at end of file" annotates the previous
        # row and belongs to neither side's count.
        if row.startswith("\\"):
            continue
        if row.startswith("+"):
            if path is not None:
                anchors.setdefault(path, set()).add(new_line)
            new_line += 1
            new_remaining -= 1
        elif row.startswith("-"):
            old_remaining -= 1
        else:
            new_line += 1
            new_remaining -= 1
            old_remaining -= 1

    return anchors


def extract_findings(response: str) -> list:
    """Pull the findings array out of the reviewer's text response.

    The prompt asks for a bare fenced block and nothing else, but a stray
    sentence either side is the likeliest way for the model to drift, and
    re-prompting costs another full model call.
    """
    match = FENCED_ARRAY.search(response)
    raw = match.group(1) if match else None
    if raw is None and response.strip().startswith("["):
        raw = response.strip()
    if raw is None:
        raise ReviewerOutputError("response contained no JSON findings array")

    try:
        findings = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReviewerOutputError(
            f"findings block is not valid JSON: {exc}"
        ) from exc

    if not isinstance(findings, list):
        raise ReviewerOutputError("findings must be a JSON array")
    return findings


def _coerce_line(value: object) -> int | None:
    """Read a line number, accepting 42 and "42" but nothing lossy.

    `bool` is an `int` subclass, so a stray `"line": true` would otherwise
    become line 1 — a real anchor on an unrelated line.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def build_comments(
    findings: list, anchors: dict[str, set[int]]
) -> tuple[list[dict], list[str]]:
    """Convert findings into review comments, dropping unpostable ones.

    Returns the comments and a reason per dropped finding. Dropping beats
    failing: one bad anchor would otherwise cost the whole review.
    """
    comments: list[dict] = []
    skipped: list[str] = []

    for finding in findings:
        if not isinstance(finding, dict):
            skipped.append(f"not a JSON object: {finding!r}")
            continue

        path = _resolve_path(str(finding.get("path") or "").strip(), anchors)
        body = str(finding.get("body") or "").strip()
        line = _coerce_line(finding.get("line"))

        if line is None:
            skipped.append(f"{path or '<no path>'}: line is not a whole number")
            continue
        if not path or not body:
            skipped.append(f"{path or '<no path>'}:{line}: empty path or body")
            continue
        if line not in anchors.get(path, frozenset()):
            skipped.append(f"{path}:{line}: not a line this PR adds")
            continue

        comments.append(
            {"path": path, "line": line, "side": "RIGHT", "body": body}
        )

    return comments, skipped


def build_parser() -> argparse.ArgumentParser:
    """The CLI, exposed so a test can pin it against the workflow's call."""
    parser = argparse.ArgumentParser(
        description="Build a GitHub review payload from AI reviewer findings."
    )
    parser.add_argument(
        "--result",
        required=True,
        type=Path,
        help="agy result file (--output-format json)",
    )
    parser.add_argument(
        "--diff",
        required=True,
        type=Path,
        help="the exact diff the reviewer was shown",
    )
    parser.add_argument(
        "--label", required=True, help="review type, e.g. Correctness"
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="payload destination; written only when there is something to post",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        result = json.loads(args.result.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return report_infra_fault(
            infra_fault(
                CHECKER, f"cannot read reviewer result {args.result}: {exc}"
            )
        )
    if not isinstance(result, dict):
        return report_infra_fault(
            infra_fault(
                CHECKER, f"reviewer result {args.result} is not a JSON object"
            )
        )

    response = result.get("response") or ""
    try:
        findings = extract_findings(response)
    except ReviewerOutputError as exc:
        print("Reviewer response was:")
        print(response[:2000])
        return report_infra_fault(
            infra_fault(CHECKER, f"reviewer output unusable: {exc}")
        )

    # errors="replace", because the workflow trims the diff to a byte budget
    # with `head -c` and that cut lands inside a multi-byte character sooner
    # or later — any diff touching an em dash or an accent is a candidate.
    # Strict decoding turned that into an uncaught UnicodeDecodeError that
    # threw away the entire review. Nothing here needs the mangled bytes:
    # anchors are computed from line structure and ASCII prefixes, and comment
    # bodies come from the model, not from the diff.
    try:
        diff = args.diff.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return report_infra_fault(
            infra_fault(CHECKER, f"cannot read diff {args.diff}: {exc}")
        )

    comments, skipped = build_comments(findings, added_line_anchors(diff))
    # A plain log line, not a ::warning:: annotation. Which findings were
    # dropped is debugging detail for whoever is looking at this job, and
    # nothing a contributor reading their PR could act on.
    for reason in skipped:
        print(f"  dropped finding — {reason}")
    print(f"{len(findings)} finding(s) returned, {len(comments)} postable.")

    if not comments:
        return EXIT_OK

    # `body` is required by the REST API whenever `event` is COMMENT.
    payload = {
        "event": "COMMENT",
        "body": f"Automated **{args.label}** review — {len(comments)} finding(s).",
        "comments": comments,
    }
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    return EXIT_OK


if __name__ == "__main__":
    # guard(): an unhandled exception must surface as a CI fault naming this
    # checker, not as a bare traceback under a "problem with your PR" banner.
    sys.exit(guard(CHECKER, main))
