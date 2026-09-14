#!/usr/bin/env python3
"""
Trim a PR diff to what is worth reviewing, and size the review from it.

Used by .github/workflows/_ai-pr-review-core.yml, between fetching the diff
and building the reviewer's prompt.

Two jobs, both of which exist because the reviewer's prompt has a hard byte
budget (Linux caps one argv string at 131072 bytes, so the workflow allows
125000 and truncates the diff to fit).

1. DROP WHAT NOBODY REVIEWS. A regenerated `uv.lock` is thousands of lines
   that no reviewer reads and that no comment can usefully land on, and every
   one of those lines displaces a line of hand-written code from the prompt.
   On a PR that mixes the two, the lockfile wins purely by being longer, and
   the code it was generated from never reaches the model at all.

2. SIZE THE REVIEW. The reviewer was previously told nothing about how many
   findings a PR of this size should yield, and settled at one or two
   regardless. The budget below is computed from REVIEWABLE churn, so a PR
   that is 1400 lines of lockfile plus 80 lines of code is sized as the small
   PR it is.

Both numbers are deterministic, which is the point: a model asked to judge
its own thoroughness will not, and a model asked to ignore a lockfile it can
see in its context still spends attention on it.

Usage:
  python3 prepare_review_diff.py \
    --diff pr_diff.txt \
    --out pr_diff_reviewable.txt \
    --github-output "${GITHUB_OUTPUT}"

Outputs (to --github-output, or stdout when it is absent):
  reviewable        `false` when nothing is left to review
  reviewable_lines  added+removed lines across the kept files
  budget            comments expected from ONE lane
  kept_files        files surviving the filter
  skipped_files     files dropped, with a one-word reason in the log

Exit codes:
  0  filtered diff written (even when it is empty — `reviewable=false` says so)
  2  CI fault — the diff this workflow just fetched could not be read
"""

import argparse
import math
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

CHECKER = "prepare_review_diff.py"

# What never earns a review comment: (pattern, reason), matched
# case-insensitively against the full path. Kept deliberately in step with
# .agents/skills/github-pr-review/scripts/plan_review.py, which makes the same
# judgement for the interactive reviewer.
SKIP_PATTERNS = [
    (
        r"(^|/)(package-lock\.json|pnpm-lock\.yaml|yarn\.lock|poetry\.lock"
        r"|Cargo\.lock|Gemfile\.lock|composer\.lock|go\.sum|uv\.lock)$",
        "lockfile",
    ),
    (r"(^|/)(vendor|node_modules|third_party|external)/", "vendored"),
    (r"(^|/)(dist|build|out|target)/", "build output"),
    (r"(^|/)__snapshots__/", "snapshot"),
    (r"\.snap$", "snapshot"),
    (r"\.min\.(js|css)$", "minified"),
    (r"\.(pb|pb2)\.(go|py|js|ts|cc|h)$", "generated protobuf"),
    (r"_pb2(_grpc)?\.pyi?$", "generated protobuf"),
    (r"\.generated\.[a-z]+$", "generated"),
    (r"(^|/)generated/", "generated"),
    (
        r"\.(png|jpe?g|gif|svg|ico|webp|pdf|woff2?|ttf|eot|zip|tar|gz|jar"
        r"|so|dylib|dll)$",
        "binary asset",
    ),
    (r"(^|/)testdata/", "test fixture"),
]

# A data file this large is a dump whatever it is called. Small ones are
# hand-written config and very much worth reviewing, so the size is doing the
# work here, not the extension.
BULK_DATA_EXT = {
    ".json",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".xml",
    ".sql",
    ".txt",
    ".ndjson",
    ".jsonl",
}
BULK_DATA_CHURN = 500

# How many THROTTLED lanes run against one PR. Used only to divide the budget
# below; the lanes themselves never learn about each other. The deterministic
# house-rules lane is exempt from the budget and is not counted here.
# `test_review_budget.py` pins this against policy.yml's `lanes` list.
LANE_COUNT = 4

# The interactive skill budgets 2-20 comments for a whole review, scaled by
# churn, and caps it at 20. Here four lanes run as separate concurrent jobs
# and cannot coordinate, so each is given a share that keeps the total at that
# same cap however many findings the others come back with.
GLOBAL_CAP = 20

# (max_reviewable_churn, comments expected from one lane).
BUDGET_TABLE = [
    (50, 2),
    (200, 3),
    (math.inf, GLOBAL_CAP // LANE_COUNT),
]

FILE_HEADER = re.compile(r"^diff --git ")
# The b-side of a `diff --git` header. Quoted when the path holds a space or a
# non-ASCII byte, which git escapes C-style.
GIT_HEADER_PATHS = re.compile(r'^diff --git (?:"?a/(.+?)"?) (?:"?b/(.+?)"?)$')


def budget_for(churn: int) -> int:
    """Comments one lane is expected to produce for this much churn."""
    for ceiling, budget in BUDGET_TABLE:
        if churn <= ceiling:
            return budget
    return BUDGET_TABLE[-1][1]


def skip_reason(path: str, churn: int) -> str | None:
    """Why this file earns no review comment, or None if it does."""
    for pattern, reason in SKIP_PATTERNS:
        if re.search(pattern, path, re.IGNORECASE):
            return reason
    suffix = Path(path).suffix.lower()
    if suffix in BULK_DATA_EXT and churn >= BULK_DATA_CHURN:
        return "bulk data"
    return None


def _section_path(section: list[str]) -> str | None:
    """The new-side path a diff section is about, or None when it is deleted.

    `+++ b/x` is preferred because it is where the reviewer's own findings are
    anchored. It is absent for a deletion (`+++ /dev/null`) and for a
    rename or mode change carrying no hunks, so the `diff --git` header is the
    fallback — it is the only line every section is guaranteed to have.
    """
    for row in section:
        if row.startswith("+++ "):
            target = row[4:].strip()
            if target == "/dev/null":
                return None
            return target[2:] if target.startswith(("a/", "b/")) else target
    header = GIT_HEADER_PATHS.match(section[0]) if section else None
    if header:
        return header.group(2)
    return None


def _section_churn(section: list[str]) -> int:
    """Added plus removed lines, ignoring the `---`/`+++` file headers."""
    churn = 0
    for row in section:
        if row.startswith(("+++ ", "--- ")):
            continue
        if row.startswith(("+", "-")):
            churn += 1
    return churn


def split_sections(diff: str) -> tuple[list[str], list[list[str]]]:
    """(preamble, per-file sections) — sections start at `diff --git`.

    A diff with no `diff --git` at all is returned whole as the preamble, so
    an unexpected format degrades to "review everything" rather than to
    "review nothing". Silently discarding a diff we failed to parse would
    turn a format change into a green check on an unreviewed PR.
    """
    preamble: list[str] = []
    sections: list[list[str]] = []
    current: list[str] | None = None

    for row in diff.split("\n"):
        if FILE_HEADER.match(row):
            if current is not None:
                sections.append(current)
            current = [row]
        elif current is None:
            preamble.append(row)
        else:
            current.append(row)

    if current is not None:
        sections.append(current)
    return preamble, sections


# ------------------------------------------------------- fitting the budget
#
# The prompt has a hard byte budget (agy takes it as one argv string, and
# Linux caps that at 131072). Until this existed the workflow met the budget
# with `head -c`, which cuts at a byte offset in git's own order -- roughly
# alphabetical. That is not a review policy, it is an accident, and on PR
# #2626 it produced this: 662KB of diff, ~100KB of budget, and the budget went
# to the six paths that sorted first. Four of the six were markdown. Not one
# of the fourteen source files was reviewed in full, 31 files were invisible,
# and the author was told to split the PR up.
#
# So pack the budget deliberately instead:
#
#   1. SOURCE before tests before prose. A defect in shipped code costs more
#      than one in a test, which costs more than one in documentation, and
#      when there is not room for everything that ranking is the decision.
#   2. Smallest first WITHIN a tier, which maximises how many files are seen
#      whole. One 79KB file would otherwise take four fifths of the budget.
#   3. Whole files, with at most one partial at the end. A 2.7KB slice of a
#      74KB file is not a review of anything -- an equal split across 37 files
#      sounds fairer and tells the model nothing about any of them.
#
# Reordering is safe: a hunk header carries its own line numbers, so an anchor
# does not depend on where its file sits in the diff.
TEST_PATH = re.compile(r"(^|/)tests?/|(^|/)test_[^/]*$|_test\.[a-z]+$", re.I)
PROSE_EXT = {".md", ".markdown", ".rst", ".txt", ".adoc"}

# Below this a partial file is a fragment rather than a sample, so it goes to
# the unreviewed list instead, which at least names it honestly.
MIN_PARTIAL_BYTES = 3000


def review_tier(path: str) -> int:
    """0 source, 1 test, 2 prose. Lower is packed first."""
    if Path(path).suffix.lower() in PROSE_EXT:
        return 2
    if TEST_PATH.search(path):
        return 1
    return 0


def _truncate_at_hunk(text: str, limit: int) -> str:
    """`text` cut to at most `limit` bytes, on a hunk boundary where possible.

    `head -c` cut mid-line, so the last file the model saw ended in half a
    statement and it had to guess the rest. Cutting at a `@@` keeps every hunk
    it does see intact.
    """
    kept: list[str] = []
    size = 0
    for row in text.split("\n"):
        cost = len(row.encode()) + 1
        if size + cost > limit:
            break
        kept.append(row)
        size += cost
    # Back off to the start of the last hunk, unless that discards everything.
    for index in range(len(kept) - 1, 0, -1):
        if kept[index].startswith("@@"):
            if index > 1:
                kept = kept[:index]
            break
    return "\n".join(kept)


def pack_to_budget(
    diff: str, max_bytes: int
) -> tuple[str, list[str], list[str]]:
    """(diff that fits, partially shown paths, omitted paths).

    `max_bytes <= 0` means no budget was given, and the diff is returned as
    it came. Never raises on a diff it cannot parse: an unparsed diff is
    returned whole, because a review of everything is the safe failure and a
    review of nothing looks exactly like a clean bill of health.
    """
    if max_bytes <= 0 or len(diff.encode()) <= max_bytes:
        return diff, [], []

    preamble, sections = split_sections(diff)
    if not sections:
        return diff, [], []
    head = "\n".join(preamble)
    room = max_bytes - len(head.encode()) - 1
    if room <= 0:
        return diff, [], []

    by_path = [
        (_section_path(s) or "<unknown>", "\n".join(s)) for s in sections
    ]
    order = sorted(
        range(len(by_path)),
        key=lambda i: (
            review_tier(by_path[i][0]),
            len(by_path[i][1].encode()),
            by_path[i][0],
        ),
    )

    chosen: dict[int, str] = {}
    partial: list[str] = []
    omitted: list[str] = []
    for i in order:
        path, text = by_path[i]
        cost = len(text.encode()) + 1
        if cost <= room:
            chosen[i] = text
            room -= cost
        elif room >= MIN_PARTIAL_BYTES and not partial:
            cut = _truncate_at_hunk(text, room - 64)
            if len(cut.encode()) >= MIN_PARTIAL_BYTES:
                chosen[i] = f"{cut}\n[... this file was truncated here ...]"
                partial.append(path)
                room = 0
            else:
                omitted.append(path)
        else:
            omitted.append(path)

    # Emit in the diff's ORIGINAL order. The packing decides what is shown;
    # the model should still read the PR in the shape git describes it.
    rows = [head] if head else []
    rows.extend(chosen[i] for i in sorted(chosen))
    out = "\n".join(rows).strip("\n")
    return (out + "\n" if out else ""), partial, omitted


def filter_diff(diff: str) -> tuple[str, dict]:
    """Drop unreviewable file sections. Returns (diff, stats)."""
    preamble, sections = split_sections(diff)
    kept: list[list[str]] = []
    skipped: list[tuple[str, str, int]] = []
    reviewable = 0

    for section in sections:
        path = _section_path(section)
        churn = _section_churn(section)

        if path is None:
            # `_section_path` returns None for a deletion, so recover the name
            # from the `diff --git` header. Slicing off "diff --git " left the
            # raw "a/x b/x" pair in the log, which reads as a path containing
            # a space and hides which file was actually dropped.
            header = GIT_HEADER_PATHS.match(section[0]) if section else None
            skipped.append(
                (header.group(2) if header else "<unknown>", "deleted", churn)
            )
            continue
        # A section with no churn is a pure rename or a mode change. There is
        # nothing in it to comment on, and on a migration PR it can be most of
        # the diff — 65 of 123 files on #2373.
        if churn == 0:
            skipped.append((path, "no content change", 0))
            continue
        reason = skip_reason(path, churn)
        if reason:
            skipped.append((path, reason, churn))
            continue

        kept.append(section)
        reviewable += churn

    rows = list(preamble)
    for section in kept:
        rows.extend(section)
    out = "\n".join(rows).strip("\n")
    if out:
        out += "\n"

    return out, {
        "reviewable_lines": reviewable,
        "kept_files": len(kept),
        "skipped": skipped,
    }


def build_parser() -> argparse.ArgumentParser:
    """The CLI, exposed so a test can pin it against the workflow's call."""
    parser = argparse.ArgumentParser(
        description="Trim a PR diff to the files worth reviewing."
    )
    parser.add_argument(
        "--diff", required=True, type=Path, help="the fetched PR diff"
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="filtered diff destination"
    )
    parser.add_argument(
        "--budget-ceiling",
        type=int,
        default=0,
        help="hard ceiling for this lane from review_budget.py; the budget "
        "this script prints is the smaller of that and its own churn-based "
        "number, so the figure the model aims at cannot exceed the figure "
        "the poster enforces. 0 means no ceiling",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=0,
        help="byte budget for the diff in the prompt. The script packs whole "
        "files into it, source before tests before prose; 0 means no budget "
        "and the filtered diff is written as-is",
    )
    parser.add_argument(
        "--unreviewed-out",
        type=Path,
        default=None,
        help="write the paths that did not fit here, one per line",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="$GITHUB_OUTPUT; printed to stdout when omitted",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # errors="replace" for the same reason post_review_comments.py needs it:
    # nothing here depends on the bytes of any individual line, and a diff
    # touching an em dash must not take the whole review down.
    try:
        diff = args.diff.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return report_infra_fault(
            infra_fault(CHECKER, f"cannot read diff {args.diff}: {exc}")
        )

    filtered, stats = filter_diff(diff)
    filtered, partial, omitted = pack_to_budget(filtered, args.max_bytes)

    try:
        args.out.write_text(filtered, encoding="utf-8")
    except OSError as exc:
        return report_infra_fault(
            infra_fault(CHECKER, f"cannot write {args.out}: {exc}")
        )

    if args.unreviewed_out:
        try:
            args.unreviewed_out.write_text(
                "".join(f"{p}\n" for p in omitted), encoding="utf-8"
            )
        except OSError as exc:
            return report_infra_fault(
                infra_fault(
                    CHECKER, f"cannot write {args.unreviewed_out}: {exc}"
                )
            )

    for path, reason, churn in stats["skipped"]:
        print(f"  skipped {path} ({reason}, {churn} lines)")
    for path in partial:
        print(f"  partial {path} (did not fit whole)")
    if omitted:
        print(
            f"  {len(omitted)} file(s) did not fit the "
            f"{args.max_bytes}-byte prompt budget:"
        )
        for path in omitted[:20]:
            print(f"    unreviewed: {path}")

    lines = stats["reviewable_lines"]
    budget = budget_for(lines)
    # Later rounds shrink. Telling the model to aim for 5 while the poster
    # will keep 1 wastes four findings and makes the log a puzzle.
    if args.budget_ceiling > 0:
        budget = min(budget, args.budget_ceiling)
    reviewable = "true" if stats["kept_files"] and lines else "false"
    print(
        f"{stats['kept_files']} file(s) / {lines} lines reviewable, "
        f"{len(stats['skipped'])} file(s) skipped; "
        f"budget {budget} comment(s) per lane."
    )

    outputs = "\n".join(
        [
            f"reviewable={reviewable}",
            f"reviewable_lines={lines}",
            f"budget={budget}",
            f"kept_files={stats['kept_files']}",
            f"skipped_files={len(stats['skipped'])}",
        ]
    )
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(outputs + "\n")
    else:
        print(outputs)

    return EXIT_OK


if __name__ == "__main__":
    # guard(): an unhandled exception must surface as a CI fault naming this
    # checker, not as a bare traceback under a "problem with your PR" banner.
    sys.exit(guard(CHECKER, main))
