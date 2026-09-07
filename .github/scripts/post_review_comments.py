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

It is also where the reviewer's output is checked rather than trusted. The
prompt's severity gate ("only critical or high") used to be the only thing
standing between a wrong finding and a contributor's PR, and it worked by
making the reviewer say almost nothing — across eight recent PRs the three
lanes posted one or two comments between them, all from Correctness. Trading
that gate for a wider one is only safe if something downstream can tell a real
finding from an invented one, so four checks live here:

  WINDOW      Each finding quotes the diff lines it sits on. Those are matched
              against the diff itself. No match means the finding was
              fabricated and it is dropped; a match a few lines off means the
              finding is real and only its arithmetic was wrong, so the anchor
              is corrected instead of the finding being lost.
  CHEAPNESS   A finding whose own stated verification procedure admits it
              needs a second file, or a traced value, or an assumed input,
              costs the author more to check than it is worth. Dropped.
  DUPLICATES  Anything already said on this PR — by an earlier run of this
              lane, by one of the other three, or by a human — is suppressed.
              Four lanes re-running on every push otherwise repeat themselves.
  CONTEXT     A verified finding on a line the PR does not add cannot be an
              inline comment, but it is still true. It goes in the review body
              rather than into the log where nobody reads it.

Usage:
  python3 post_review_comments.py \
    --result agy_result.json \
    --diff pr_diff_used.txt \
    --label Correctness \
    --out review_payload.json \
    [--repo owner/name --pr 123]

--repo/--pr enable duplicate suppression; without them the other three checks
still run. --out is written ONLY when there is something to post, so the caller
can treat "file absent" as "nothing to post".

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
import itertools
import json
import re
import subprocess
import sys
import unicodedata
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

# The fenced block, captured whole from its opening "[" to the closing
# fence. Regex cannot find where the array ends: a "]" inside a comment body
# is indistinguishable from the one that closes it, and anchoring on the LAST
# "]" in the block swallowed anything the model appended after the array.
# json's own parser draws that boundary in extract_findings instead.
FENCED_BLOCK = re.compile(r"```(?:json)?\s*(\[.*?)```", re.DOTALL)

# A window row: "  42: os.system(cmd)". Both ":" and "|" are seen as the
# separator, and the leading whitespace is the model aligning its numbers.
WINDOW_LINE = re.compile(r"^\s*(\d+)\s*[:|]\s?(.*)$")

# The keys a finding is built from. Used to find where a string value ENDS
# when the model has left raw quotes inside it — see _repair_string_values.
FINDING_KEYS = "path|line|body|window|verify_steps"

# The opener of a string-valued field: '"window": "'.
STRING_FIELD_OPEN = re.compile(rf'"(?:{FINDING_KEYS})"\s*:\s*"')

# Where such a value ends: a quote followed by the next key, or by the end of
# the object. Anything else that looks like a terminator is part of the value.
STRING_FIELD_END = re.compile(rf'"(?=\s*(?:,\s*"(?:{FINDING_KEYS})"\s*:|\}}))')

# How much work to spend reading a malformed block before giving up on it.
# The choices multiply per field, so this is a ceiling on a combinatorial walk.
# Proving a reading UNIQUE means exhausting the walk, so the cap has to clear
# a real block with room to spare: the three-finding block in
# fixtures/malformed_findings_response.txt spends 6091 of it and the
# two-finding #2566 block 383. Exhausting the cap is treated as "ambiguous",
# never as "unique" — see _repaired_findings.
MAX_REPAIR_STEPS = 16384

# How many candidate ends to weigh for a single field. Every later field
# boundary is a candidate end for an earlier field, so this is what stops a
# many-finding block fanning out by its own width. The correct end is the
# first in every real case seen; the rest are only there to expose ambiguity,
# and a reading only reachable past the eighth is one this declines to make.
MAX_ENDS_PER_FIELD = 8

# String fields past which a block is not a findings array a review produced,
# and is not worth reading. A lane is asked for about five findings of four
# string fields each; the two real blocks here carry 8 and 12. Checked before
# the walk starts, because the walk holds a copy of the reading so far per
# branch: a runaway thousand-finding block cost 400MB of them before this.
MAX_REPAIR_FIELDS = 64

# Literal escape text a model writes where the diff holds the real character.
ESCAPE_SEQ = re.compile(
    r"\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}|\\x[0-9a-fA-F]{2}"
)

# Phrases in verify_steps that admit the reader must leave the anchored lines.
# Ported from .agents/skills/github-pr-review/scripts/verify_findings.py, where
# they were tuned against two real reviews.
NOT_CHEAP_MARKERS = re.compile(
    r"\btrace\b|\bassum\w*\b|consider the case|if an attacker|"
    r"\bsimulat\w*\b|\bimagine\b|run the code|execute\b|another file|"
    r"\bgrep the repo\b|across (the )?(repo|codebase)",
    re.IGNORECASE,
)

# SECURITY (b/555419958). A finding's `body` is written by a model whose only
# input is the PR diff, and on a fork PR every byte of that diff is chosen by
# its author. The body is then posted verbatim to a public pull request through
# the API, where GitHub's log masking does not reach. So the body is an
# attacker-influenced string on a public channel, and something has to say what
# shape it is allowed to be.
#
# These two limits say it. They are NOT the boundary — the empty agy tool
# allowlist in the workflow is, and the credential scan there is the precise
# check. What they do is make the channel too narrow to carry a credential
# without the model first chopping it up, which is the difference between an
# exfiltration that works on the first try and one that has to be engineered.
# Treat them as a speed bump with a second job, not as a proof.
#
# The second job is the honest one: the prompt asks for "1-2 plain sentences"
# and nothing enforced it. A model that returns an essay is misbehaving whether
# or not anyone is attacking, and a review comment nobody will read is not
# worth posting either.
#
# 600 characters is roughly 100 words — four or five sentences, well clear of
# the two the prompt asks for, and clear too of a grouped finding that has to
# say how many instances it covers.
MAX_BODY_CHARS = 600

# The longest unbroken non-whitespace run a body may contain. Calibrated
# against real data, not guessed: the longest path in this repository is 123
# characters (java/agents/time-series-forecasting/...ForecastingAgent.java),
# and quoting a path is exactly what a legitimate finding does. So the cap has
# to clear that with room, or the check costs real reviews. 160 does, while
# still refusing a credential pasted in one piece — the ADC file this runner
# holds is ~1KB and its embedded token alone runs past 200.
MAX_UNBROKEN_RUN = 160

# How far an anchor may be wrong before a window match stops being believable.
MAX_DRIFT = 3

# Below this many characters, a window line must equal the diff line exactly.
# A short line is a prefix of almost anything and matches spuriously.
MIN_PREFIX = 8

# A duplicate is the same line or within this many of it. Tight on purpose:
# widening it starts eating genuinely new findings near an old comment.
PROXIMITY = 2

# Token overlap above which two comments are saying the same thing.
SIMILARITY = 0.55

_STOPWORDS = set(
    """a an the is are was were be been being this that these those it its of to
    in on for with from by at as and or not no any some all each every into out
    here there which what when where line lines file files code value values
    never only still also just even than then them they their should does did
    has have""".split()
)


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


def walk_right_side(
    diff: str,
) -> tuple[dict[str, set[int]], dict[str, dict[int, str]]]:
    """Read the diff's RIGHT side once: (added anchors, line text).

    The first return value is the set of new-file lines each file ADDS, which
    is what GitHub will accept as an inline anchor. The second is the text of
    every new-file line the diff shows — added and context alike — which is
    what a finding's quoted window is checked against.

    Both come from one walk because they are the same traversal, and two
    traversals that disagree about where a hunk starts would put a comment on
    a line whose text was read from somewhere else.

    Only `+` lines are valid anchors on the RIGHT side of a review, so the
    first value doubles as enforcement of the prompt's "added lines only"
    rule. Context lines are still worth indexing: a finding anchored to one is
    real but unpostable inline, and belongs in the review body.

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
    text: dict[str, dict[int, str]] = {}
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
                text.setdefault(path, {})[new_line] = row[1:]
            new_line += 1
            new_remaining -= 1
        elif row.startswith("-"):
            old_remaining -= 1
        else:
            if path is not None:
                text.setdefault(path, {})[new_line] = row[1:]
            new_line += 1
            new_remaining -= 1
            old_remaining -= 1

    return anchors, text


def added_line_anchors(diff: str) -> dict[str, set[int]]:
    """The new-file lines each path adds. See walk_right_side."""
    return walk_right_side(diff)[0]


def extract_findings(response: str) -> list:
    """Pull the findings array out of the reviewer's text response.

    The prompt asks for a bare fenced block and nothing else, but a stray
    sentence either side is the likeliest way for the model to drift, and
    re-prompting costs another full model call.

    The inside of the block drifts too: a reviewer answered PR #2547 with
    "[]\\n[]" — two arrays in one fence — which is not a JSON document, so
    parsing the block as one value failed the run as a CI fault over a review
    that had simply found nothing. Each top-level array is decoded in turn and
    their findings concatenated, and a decode failure after the first array
    has parsed ends the scan rather than the review, because trailing prose is
    drift and not a reason to throw away findings already in hand. Only the
    first array failing to parse is fatal — at that point there is nothing to
    post and nothing to infer.

    The LAST fenced block wins, not the first. The prompt now asks the
    reviewer to work through the diff in plain text before answering, because
    a model told to emit nothing but JSON does no reasoning and finds
    correspondingly little. That scan quotes diff rows and sometimes brackets
    them, so the first fenced array in the response is no longer reliably the
    answer — the prompt says the findings block comes last, and this reads it
    from the same end.

    A block that will not parse AT ALL is repaired, and failing that salvaged
    finding by finding, rather than thrown away whole. Findings quote source
    verbatim in `window`, and source is full of double quotes: a reviewer
    emitted

        "window": "  211:     assert res["success"] is True\\n ..."

    which is one unescaped pair away from valid and killed an entire review
    of three good findings.

    Repair runs first because it recovers the WHOLE block, where salvage keeps
    only the findings that happened to be well-formed already. Salvage alone
    is not enough: the same quoting habit corrupts every window drawn from the
    same file, so on a quote-dense diff there are no well-formed siblings left
    to keep. Repair declines anything it cannot read one single way, so salvage
    remains the fallback for a block that is malformed in some other way, and
    one malformed finding still costs only that finding.
    """
    matches = list(FENCED_BLOCK.finditer(response))
    # Both branches guarantee a block starting at "[", so the scan below
    # always decodes at least once.
    block = matches[-1].group(1) if matches else None
    if block is None and response.strip().startswith("["):
        block = response.strip()
    if block is None:
        raise ReviewerOutputError("response contained no JSON findings array")

    decoder = json.JSONDecoder()
    findings: list = []
    parsed_any = False
    index = 0

    while (start := block.find("[", index)) != -1:
        try:
            array, index = decoder.raw_decode(block, start)
        except json.JSONDecodeError as exc:
            if parsed_any:
                break
            # Repair before salvage: it recovers every finding in the block,
            # where salvage keeps only the ones that were already well-formed.
            repaired = _repaired_findings(block)
            if repaired is not None:
                print(
                    f"  findings block is malformed JSON ({exc}); "
                    f"repaired {len(repaired)} finding(s) in it"
                )
                return _dedupe(repaired)
            salvaged = _salvage_findings(block, decoder)
            if salvaged:
                print(
                    f"  findings block is malformed JSON ({exc}); "
                    f"salvaged {len(salvaged)} finding(s) from it"
                )
                return _dedupe(salvaged)
            raise ReviewerOutputError(
                f"findings block is not valid JSON: {exc}"
            ) from exc

        parsed_any = True
        findings.extend(array)

    return _dedupe(findings)


def _dedupe(findings: list) -> list:
    """A repeated block repeats its findings, and posting the same note twice
    on the same line is worse than posting it once."""
    seen: set[str] = set()
    unique: list = []
    for finding in findings:
        key = json.dumps(finding, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique


def _escape_value(value: str) -> str:
    """Re-escape a value's double quotes, normalising first so it is idempotent."""
    return value.replace('\\"', '"').replace('"', '\\"')


def _repair_readings(block: str, budget: list[int]):
    """Every way of escaping the block, one per choice of where values end.

    A value's end is looked for from the SCHEMA — the quote that precedes the
    next known key, or the one that closes the object — because the obvious
    cheap rule, "a quote followed by `,`, `]` or `}`", is wrong on the very
    input this exists for: `[ -f "yarn.lock" ]` puts a `]` right after the
    stray quote and would end the array mid-string.

    The schema rule is not unambiguous either, so the first MAX_ENDS_PER_FIELD
    candidate ends are branched on rather than just the first, and the caller
    decides between the readings they produce.

    Depth-first over an explicit stack, not recursion: one field is one frame,
    and a runaway model answering with hundreds of findings would exceed the
    interpreter's limit. That mattered — a RecursionError escapes the caller
    entirely, so a block salvage could have read went to a CI fault instead.
    `budget` is charged for every reading completed and every branch taken, so
    the walk is capped whatever shape it takes: the combinations multiply per
    field, and a block nobody can read is not worth an unbounded search.
    """
    stack: list[tuple[int, str]] = [(0, "")]

    while stack:
        if budget[0] <= 0:
            return
        index, prefix = stack.pop()

        opener = STRING_FIELD_OPEN.search(block, index)
        if opener is None:
            budget[0] -= 1
            yield prefix + block[index:]
            continue

        head = prefix + block[index : opener.end()]
        ends = list(
            itertools.islice(
                STRING_FIELD_END.finditer(block, opener.end()),
                MAX_ENDS_PER_FIELD,
            )
        )
        if not ends:
            # No end to find, so there is nothing to repair past this point.
            budget[0] -= 1
            yield head + block[opener.end() :]
            continue

        # Reversed, so the earliest end is popped first and the cheapest
        # reading is the one a budget-limited walk is most likely to reach.
        #
        # Each push copies the reading so far, so pushes are charged to the
        # budget as well. Counting only the steps that reached a leaf let a
        # wide block spend minutes building readings it never finished: every
        # later field boundary is a candidate end for every earlier field, so
        # a 400-finding block pushed ~1600 copies per step and ran for over
        # two minutes before this.
        for end in reversed(ends):
            budget[0] -= 1
            value = block[opener.end() : end.start()]
            stack.append((end.end(), f'{head}{_escape_value(value)}"'))


def _repaired_findings(block: str) -> list | None:
    """The findings a malformed block yields, when exactly one reading fits.

    Findings quote source verbatim in `window`, and source is full of double
    quotes, so the model regularly emits

        "window": "  252:   elif [ -f "yarn.lock" ]; then\\n ..."

    which is not JSON. `_salvage_findings` cannot help: it drops a finding it
    cannot decode, and one quoting habit corrupts EVERY window drawn from the
    same file, so a quote-dense diff loses the whole review rather than one
    finding of it. That is what happened on #2566, where both findings quoted
    shell out of a workflow file and the run died as a CI fault.

    Escaping alone is not enough, because where a value ENDS can be genuinely
    ambiguous. A window that quotes findings-shaped source — which this repo's
    own tests contain — offers a second reading in which the value stops early
    and the source's own `"line":` and `"body":` become the finding's fields:

        "window": "  919: [{"path": "a.py", "line": 1, "body": "real"}]"

    reads just as well as a finding anchored at line 1 with the body "real".
    That parses, so parsing cannot be the test. Posting source scraped out of
    a window as though it were a review comment is worse than posting nothing,
    so a block is repaired ONLY when exactly one reading survives; anything
    else falls through to salvage and, failing that, to the CI fault. The two
    real regressions this exists for both have exactly one reading.
    """
    fields = sum(1 for _ in STRING_FIELD_OPEN.finditer(block))
    if fields > MAX_REPAIR_FIELDS:
        return None

    readings: dict[str, list] = {}
    decoder = json.JSONDecoder()
    budget = [MAX_REPAIR_STEPS]

    for candidate in _repair_readings(block, budget):
        try:
            array, _ = decoder.raw_decode(candidate, candidate.find("["))
        except json.JSONDecodeError:
            continue
        if isinstance(array, list) and array:
            readings.setdefault(json.dumps(array, sort_keys=True), array)
            if len(readings) > 1:
                return None

    # An exhausted budget means readings beyond the ones seen may exist, so
    # "exactly one" has not actually been established.
    if budget[0] <= 0 or len(readings) != 1:
        return None
    return next(iter(readings.values()))


def _salvage_findings(block: str, decoder: json.JSONDecoder) -> list[dict]:
    """Decode findings one object at a time, skipping ones that will not parse.

    Only objects carrying the fields a finding must have are kept. Advancing
    past a failure means the next `{` tried may be one inside a string, so
    something has to reject the resulting debris; requiring `path` and `body`
    does, and every real finding has both.
    """
    salvaged: list[dict] = []
    index = 0
    while (start := block.find("{", index)) != -1:
        try:
            candidate, index = decoder.raw_decode(block, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if (
            isinstance(candidate, dict)
            and str(candidate.get("path") or "").strip()
            and str(candidate.get("body") or "").strip()
        ):
            salvaged.append(candidate)
    return salvaged


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


def _norm(text: str) -> str:
    """Reduce a source line to a comparable ASCII skeleton.

    A model writes an emoji as the literal escape text `\\u26a0` where the diff
    holds the real character, and round-tripping through unicode_escape does
    not reconcile the two — it mojibakes the diff side instead, so every line
    holding a non-ASCII character then looks fabricated.

    Comparing ASCII skeletons sidesteps that: drop escape sequences, drop
    non-ASCII, drop whitespace. Real fabrication still differs in the ASCII
    text, which is where the substance of a line of code lives.
    """
    text = ESCAPE_SEQ.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if c.isascii() and not c.isspace())


def _window_rows(window: object) -> list[tuple[int, str]]:
    """[(claimed_line, text)] from a finding's quoted window."""
    rows: list[tuple[int, str]] = []
    for raw in str(window or "").split("\n"):
        match = WINDOW_LINE.match(raw)
        if not match:
            continue
        # Models abbreviate a long line with an ellipsis. A Unicode one
        # vanishes in _norm as non-ASCII, but an ASCII "..." survives and
        # would turn a valid prefix into a mismatch.
        rows.append(
            (
                int(match.group(1)),
                re.sub(r"(\.{3}|\u2026)\s*$", "", match.group(2)),
            )
        )
    return rows


def _window_matches(
    rows: list[tuple[int, str]], lines: dict[int, str], offset: int
) -> int:
    """How many window rows match the diff at this offset; 0 if any conflicts.

    A row whose line the diff does not show is not a conflict — the model may
    quote a couple of lines either side of a hunk boundary — but it does not
    count towards the match either.
    """
    matched = 0
    for number, claimed in rows:
        actual = lines.get(number + offset)
        if actual is None:
            continue
        want, have = _norm(claimed), _norm(actual)
        if not want:
            continue
        if want == have:
            matched += 1
        elif min(len(want), len(have)) >= MIN_PREFIX and (
            want.startswith(have) or have.startswith(want)
        ):
            # Prefix, not substring: a one-character line is a substring of
            # almost anything and matches spuriously.
            matched += 1
        else:
            return 0
    return matched


def _snap_to_window(line: int, rows: list[tuple[int, str]], offset: int) -> int:
    """Pull an anchor that sits outside its own verified window back into it.

    The prompt requires the anchored line to be one of the window's rows, so
    the two disagreeing means the model quoted the right code and then named a
    different line beside it. The window has been checked against the diff and
    the anchor has not, so the verified side wins.

    Without this the comment lands a line or two from the thing it is about —
    a placeholder value flagged on the `return` statement underneath it, say —
    which reads as carelessness even when the finding itself is right.
    """
    claimed = [number + offset for number, _text in rows]
    if line in claimed:
        return line
    return min(
        claimed, key=lambda candidate: (abs(candidate - line), candidate)
    )


def implausible_body(body: str) -> str | None:
    """Why this body is not shaped like a review comment, or None if it is.

    See MAX_BODY_CHARS. Deliberately says nothing about what a body may
    CONTAIN: a keyword deny-list ("external_account", "Bearer", ...) reads as
    security and is not, because the thing being filtered is written by a model
    that the same input can instruct to reword, encode or spell out whatever
    the list names. Shape is the property an attacker cannot negotiate away —
    a credential is long, or it is chopped into pieces small enough that
    reassembling it is its own problem.
    """
    if len(body) > MAX_BODY_CHARS:
        return (
            f"body is {len(body)} chars, over the {MAX_BODY_CHARS}-char limit"
        )

    longest = max((len(run) for run in body.split()), default=0)
    if longest > MAX_UNBROKEN_RUN:
        return (
            f"body contains a {longest}-char unbroken run, over the "
            f"{MAX_UNBROKEN_RUN}-char limit"
        )

    return None


def check_window(
    finding: dict, line: int, lines: dict[int, str]
) -> tuple[bool, int, str]:
    """(verified, line, reason) for a finding's quoted window.

    A window matching at a CONSISTENT offset is arithmetic drift, not
    fabrication: the finding is real and only the model's line counting was
    wrong, so the anchor is corrected rather than the finding thrown away.
    Requiring two matching rows before accepting a shift keeps a single
    coincidental match from moving a comment onto unrelated code.

    Counting new-file line numbers out of a unified diff by hand is the part
    of this job a model is worst at, and it is the part that decides whether a
    correct finding is postable at all. Both repairs here exist because
    dropping a finding over its arithmetic throws away work that was right.
    """
    rows = _window_rows(finding.get("window"))
    if not rows:
        return False, line, "no window supplied"

    if _window_matches(rows, lines, 0):
        snapped = _snap_to_window(line, rows, 0)
        if snapped != line:
            return (
                True,
                snapped,
                f"anchor {line} was outside its own window; moved to {snapped}",
            )
        return True, line, "window matches the diff"

    for step in range(1, MAX_DRIFT + 1):
        for offset in (step, -step):
            if _window_matches(rows, lines, offset) >= 2:
                return (
                    True,
                    _snap_to_window(line + offset, rows, offset),
                    f"window matched {offset:+d} lines away; anchor corrected",
                )

    number, claimed = rows[0]
    actual = lines.get(number, "")
    return (
        False,
        line,
        f"window says {claimed.strip()[:40]!r}, diff has {actual.strip()[:40]!r}",
    )


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z_][a-z_0-9]{3,}", str(text).lower())
        if word not in _STOPWORDS
    }


def fetch_existing_comments(repo: str, pr: int) -> list[dict]:
    """Everything already said on this PR, inline and top-level.

    Four lanes review every push, so without this the same observation is
    re-posted on every `synchronize` and each lane repeats whatever the other
    three found in the overlap between their remits.

    A failure here is not fatal. Suppression improves the review; it is not a
    precondition for having one, and losing a whole review to a transient API
    error is a worse outcome than posting a comment twice.

    Those failures are logged as plain lines, not `::warning::` annotations,
    for the same reason the dropped findings in main() are: a contributor
    reading their PR can neither act on this nor be helped by seeing it.
    """
    existing: list[dict] = []
    for endpoint, kind in (
        (f"repos/{repo}/pulls/{pr}/comments", "inline"),
        (f"repos/{repo}/issues/{pr}/comments", "top-level"),
    ):
        try:
            proc = subprocess.run(
                ["gh", "api", "--paginate", f"{endpoint}?per_page=100"],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"  could not read {kind} comments: {exc}")
            continue
        if proc.returncode != 0:
            print(
                f"  could not read {kind} comments: {proc.stderr.strip()[:200]}"
            )
            continue
        # --paginate concatenates one JSON array per page, so decode them in
        # sequence rather than parsing the output as a single document.
        decoder = json.JSONDecoder()
        text, index = proc.stdout, 0
        while (start := text.find("[", index)) != -1:
            try:
                batch, index = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                break
            for item in batch:
                existing.append(
                    {
                        "kind": kind,
                        "path": item.get("path"),
                        "line": item.get("line"),
                        "original_line": item.get("original_line"),
                        "body": item.get("body") or "",
                    }
                )
    return existing


def build_exclusions(
    existing: list[dict],
) -> tuple[dict[str, dict[int, dict]], list[tuple[set[str], dict]]]:
    """(line zones, tokenised bodies) from comments already on the PR."""
    zones: dict[str, dict[int, dict]] = {}
    texts: list[tuple[set[str], dict]] = []
    for comment in existing or []:
        if (comment.get("body") or "").strip():
            texts.append((_tokens(comment["body"]), comment))
        if comment.get("kind") != "inline":
            continue
        path = comment.get("path")
        if not path:
            continue
        # `line` is nulled by GitHub once a thread goes outdated, and only
        # `original_line` survives. Both are claimed so a moved comment still
        # blocks the place it was originally made about.
        for anchor in (comment.get("line"), comment.get("original_line")):
            if not anchor:
                continue
            for delta in range(-PROXIMITY, PROXIMITY + 1):
                zones.setdefault(path, {}).setdefault(anchor + delta, comment)
    return zones, texts


def already_raised(
    path: str,
    line: int,
    body: str,
    zones: dict[str, dict[int, dict]],
    texts: list[tuple[set[str], dict]],
) -> str:
    """Why this finding repeats something already on the PR, or ""."""
    if zones.get(path, {}).get(line):
        return "already commented on this line"

    mine = _tokens(body)
    if len(mine) < 4:
        return ""
    for tokens, _comment in texts:
        if len(tokens) < 4:
            continue
        if len(mine & tokens) / min(len(mine), len(tokens)) >= SIMILARITY:
            return "very similar to a comment already on this PR"
    return ""


def build_comments(
    findings: list,
    anchors: dict[str, set[int]],
    line_text: dict[str, dict[int, str]] | None = None,
    existing: list[dict] | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Sort findings into inline comments, body notes, and the discarded.

    Returns (comments, notes, skipped-with-reasons). Dropping beats failing:
    one bad anchor would otherwise cost the whole review.

    `notes` are findings whose window verified against a line the PR does not
    ADD — real, but with nowhere to hang inline. They used to go into the job
    log and be lost; they go in the review body instead. Only window-verified
    findings qualify, because an unverifiable line number is a model
    arithmetic error and promoting those would surface exactly the mistakes
    this function exists to catch.
    """
    line_text = line_text or {}
    comments: list[dict] = []
    notes: list[dict] = []
    skipped: list[str] = []
    zones, texts = build_exclusions(existing or [])

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

        # Before anything that could promote this body onto the PR — inline or
        # as a note in the review body, both of which are public.
        malformed = implausible_body(body)
        if malformed:
            skipped.append(f"{path}:{line}: {malformed}")
            continue

        steps = str(finding.get("verify_steps") or "")
        marker = NOT_CHEAP_MARKERS.search(steps)
        if marker:
            skipped.append(
                f"{path}:{line}: not cheap to verify ({marker.group(0)!r} "
                "in verify_steps)"
            )
            continue

        # Window first: it can move the anchor, and everything below depends
        # on the anchor being the one the finding is really about.
        declared = line
        verified, line, reason = check_window(
            finding, line, line_text.get(path, {})
        )
        if not verified and finding.get("window"):
            skipped.append(f"{path}:{line}: {reason}")
            continue
        if line != declared:
            print(f"  {path}: {reason}")

        duplicate = already_raised(path, line, body, zones, texts)
        if duplicate:
            skipped.append(f"{path}:{line}: {duplicate}")
            continue

        if line in anchors.get(path, frozenset()):
            comments.append(
                {"path": path, "line": line, "side": "RIGHT", "body": body}
            )
        elif verified:
            notes.append({"path": path, "line": line, "body": body})
        else:
            skipped.append(f"{path}:{line}: not a line this PR adds")

    return comments, notes, skipped


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
    parser.add_argument(
        "--repo",
        default=None,
        help="owner/name; with --pr, suppresses comments already on the PR",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="PR number; with --repo, suppresses comments already on the PR",
    )
    return parser


def build_payload(label: str, comments: list[dict], notes: list[dict]) -> dict:
    """The review payload. `body` is required whenever `event` is COMMENT."""
    header = f"Automated **{label}** review — {len(comments)} finding(s)."
    if notes:
        # These sit on lines the PR does not add, so GitHub will not take them
        # inline. Listing them here is the only way they reach the author at
        # all, and on PR #2373 this class held all three hard CI failures.
        lines = [
            header,
            "",
            "Also, on lines this PR does not change:",
            "",
        ]
        lines += [f"- `{n['path']}:{n['line']}` — {n['body']}" for n in notes]
        return {
            "event": "COMMENT",
            "body": "\n".join(lines),
            "comments": comments,
        }
    return {"event": "COMMENT", "body": header, "comments": comments}


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

    existing = (
        fetch_existing_comments(args.repo, args.pr)
        if args.repo and args.pr
        else []
    )
    if existing:
        print(f"{len(existing)} comment(s) already on this PR.")

    anchors, line_text = walk_right_side(diff)
    comments, notes, skipped = build_comments(
        findings, anchors, line_text, existing
    )
    # A plain log line, not a ::warning:: annotation. Which findings were
    # dropped is debugging detail for whoever is looking at this job, and
    # nothing a contributor reading their PR could act on.
    for reason in skipped:
        print(f"  dropped finding — {reason}")
    print(
        f"{len(findings)} finding(s) returned, {len(comments)} postable, "
        f"{len(notes)} on unchanged lines."
    )

    if not comments and not notes:
        return EXIT_OK

    args.out.write_text(
        json.dumps(build_payload(args.label, comments, notes)),
        encoding="utf-8",
    )
    return EXIT_OK


if __name__ == "__main__":
    # guard(): an unhandled exception must surface as a CI fault naming this
    # checker, not as a bare traceback under a "problem with your PR" banner.
    sys.exit(guard(CHECKER, main))
