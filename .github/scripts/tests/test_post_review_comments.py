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
"""Unit tests for post_review_comments.py.

This script decides where an automated review comment lands. Its failure
modes are quiet ones: an anchor computed one line off puts a comment on
innocent code, and a mis-parsed file header drops every finding in a file
while reporting only "not a line this PR adds". Neither shows up as a red
check — the review just says something wrong, or says nothing.

Every test below pins one of those.
"""

import json
import subprocess
import sys
from pathlib import Path

import post_review_comments as m
import pytest

# Must come after post_review_comments: importing that is what puts tools/ on
# sys.path. isort keeps plain imports above from-imports so the order holds,
# and reordering it would fail loudly at collection rather than silently.
from ci_message import EXIT_CI_FAULT

SCRIPT = Path(m.__file__)


def _diff(*rows: str) -> str:
    return "\n".join(rows) + "\n"


# --------------------------------------------------------------------------
# added_line_anchors
# --------------------------------------------------------------------------


def test_anchors_track_new_file_line_numbers():
    diff = _diff(
        "diff --git a/x.py b/x.py",
        "index 111..222 100644",
        "--- a/x.py",
        "+++ b/x.py",
        "@@ -10,3 +10,4 @@",
        " context",
        "-removed",
        "+added_one",
        "+added_two",
        " trailing",
    )
    # Hunk starts at new-file line 10: " context" is 10, the two added
    # lines are 11 and 12 (the removed line consumes no new-side number).
    assert m.added_line_anchors(diff) == {"x.py": {11, 12}}


def test_added_line_that_looks_like_a_file_header_is_not_one():
    """An added line beginning "++ " renders as the row "+++ ...".

    Prefix-sniffing read that as a `+++ b/path` header, set the path to the
    line's own text, and silently lost every later anchor in the file.
    """
    diff = _diff(
        "--- a/doc.md",
        "+++ b/doc.md",
        "@@ -1,2 +1,4 @@",
        " intro",
        "++ this documents a diff marker",
        "+a real finding lands here",
        " outro",
    )
    assert m.added_line_anchors(diff) == {"doc.md": {2, 3}}


def test_removed_line_that_looks_like_a_file_header_is_not_one():
    diff = _diff(
        "--- a/doc.md",
        "+++ b/doc.md",
        "@@ -1,3 +1,2 @@",
        " intro",
        "-- this line is going away",
        "+replacement",
    )
    assert m.added_line_anchors(diff) == {"doc.md": {2}}


def test_one_line_hunk_without_explicit_lengths():
    diff = _diff("--- a/x.py", "+++ b/x.py", "@@ -1 +1 @@", "-old", "+new")
    assert m.added_line_anchors(diff) == {"x.py": {1}}


def test_multiple_hunks_and_multiple_files():
    diff = _diff(
        "--- a/a.py",
        "+++ b/a.py",
        "@@ -1,1 +1,2 @@",
        " keep",
        "+first",
        "@@ -20,1 +21,2 @@",
        " keep",
        "+second",
        "--- a/b.py",
        "+++ b/b.py",
        "@@ -5,0 +6,1 @@",
        "+only",
    )
    assert m.added_line_anchors(diff) == {"a.py": {2, 22}, "b.py": {6}}


def test_new_file_has_no_old_side():
    diff = _diff(
        "--- /dev/null",
        "+++ b/new.py",
        "@@ -0,0 +1,2 @@",
        "+line one",
        "+line two",
    )
    assert m.added_line_anchors(diff) == {"new.py": {1, 2}}


def test_deleted_file_contributes_no_anchors():
    diff = _diff(
        "--- a/gone.py",
        "+++ /dev/null",
        "@@ -1,2 +0,0 @@",
        "-line one",
        "-line two",
    )
    assert m.added_line_anchors(diff) == {}


def test_no_newline_marker_does_not_shift_numbering():
    diff = _diff(
        "--- a/x.py",
        "+++ b/x.py",
        "@@ -1,2 +1,2 @@",
        " first",
        "-second",
        "\\ No newline at end of file",
        "+second!",
        "\\ No newline at end of file",
    )
    assert m.added_line_anchors(diff) == {"x.py": {2}}


def test_unparseable_hunk_header_yields_no_anchors():
    """Guessing a start line anchors comments onto real but wrong lines.

    A wrong anchor still passes validation and gets posted, so a hunk we
    cannot place must contribute nothing.
    """
    diff = _diff("--- a/x.py", "+++ b/x.py", "@@ garbled @@", "+added", "+more")
    assert m.added_line_anchors(diff) == {}


def test_truncated_diff_keeps_the_anchors_it_did_see():
    """The workflow cuts the diff to fit the argv budget, mid-hunk."""
    diff = (
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,9 +1,9 @@\n"
        " keep\n"
        "+added\n"
        "+part"  # cut mid-hunk, no trailing newline
        "\n[... diff truncated — review only what is shown above ...]"
    )
    assert m.added_line_anchors(diff) == {"x.py": {2, 3}}


def test_empty_diff():
    assert m.added_line_anchors("") == {}


# --------------------------------------------------------------------------
# extract_findings
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        '```json\n[{"path": "x.py"}]\n```',
        '```\n[{"path": "x.py"}]\n```',
        'Here you go:\n\n```json\n[{"path": "x.py"}]\n```\n\nHope that helps.',
        '[{"path": "x.py"}]',
        '   [{"path": "x.py"}]   ',
    ],
)
def test_extract_findings_accepts_realistic_shapes(response):
    assert m.extract_findings(response) == [{"path": "x.py"}]


def test_extract_findings_keeps_brackets_inside_a_body():
    response = '```json\n[{"body": "index arr[0] is off by one"}]\n```'
    assert m.extract_findings(response) == [
        {"body": "index arr[0] is off by one"}
    ]


def test_extract_findings_accepts_an_empty_array():
    assert m.extract_findings("```json\n[]\n```") == []


def test_extract_findings_accepts_a_repeated_empty_array():
    # What PR #2547's reviewer actually returned.
    assert m.extract_findings("```json\n[]\n[]\n```") == []


def test_extract_findings_merges_several_arrays_in_one_block():
    response = '```json\n[{"path": "x.py"}]\n[{"path": "y.py"}]\n```'
    assert m.extract_findings(response) == [{"path": "x.py"}, {"path": "y.py"}]


def test_extract_findings_drops_a_repeated_finding():
    response = '```json\n[{"path": "x.py"}]\n[{"path": "x.py"}]\n```'
    assert m.extract_findings(response) == [{"path": "x.py"}]


def test_extract_findings_keeps_findings_despite_trailing_junk():
    response = '```json\n[{"path": "x.py"}]\nand also [oops\n```'
    assert m.extract_findings(response) == [{"path": "x.py"}]


@pytest.mark.parametrize(
    "response",
    [
        "I reviewed it and found nothing.",
        '```json\n{"path": "x.py"}\n```',
        "```json\n[{oops}\n```",
        "",
    ],
)
def test_extract_findings_rejects_unusable_output(response):
    with pytest.raises(m.ReviewerOutputError):
        m.extract_findings(response)


# --------------------------------------------------------------------------
# build_comments
# --------------------------------------------------------------------------

ANCHORS = {"x.py": {10, 11}}


def test_build_comments_emits_a_right_side_line_comment():
    comments, skipped = m.build_comments(
        [{"path": "x.py", "line": 10, "body": "Off by one."}], ANCHORS
    )
    assert comments == [
        {"path": "x.py", "line": 10, "side": "RIGHT", "body": "Off by one."}
    ]
    assert skipped == []


def test_build_comments_strips_a_stray_b_prefix():
    comments, _ = m.build_comments(
        [{"path": "b/x.py", "line": 10, "body": "note"}], ANCHORS
    )
    assert comments[0]["path"] == "x.py"


def test_a_real_path_starting_with_b_slash_is_not_stripped():
    """A repo with a top-level `b/` directory has real paths like this.

    Stripping unconditionally would turn `b/pkg.py` into `pkg.py`, match
    nothing, and silently drop every finding in that directory.
    """
    anchors = {"b/pkg.py": {7}}
    comments, skipped = m.build_comments(
        [{"path": "b/pkg.py", "line": 7, "body": "note"}], anchors
    )
    assert comments[0]["path"] == "b/pkg.py"
    assert skipped == []


def test_build_comments_accepts_a_stringified_line():
    comments, _ = m.build_comments(
        [{"path": "x.py", "line": "10", "body": "note"}], ANCHORS
    )
    assert comments[0]["line"] == 10


@pytest.mark.parametrize(
    "finding",
    [
        {"path": "x.py", "line": 99, "body": "note"},  # not an added line
        {"path": "other.py", "line": 10, "body": "note"},  # untouched file
        {"path": "x.py", "line": 10, "body": "   "},  # empty body
        {"path": "", "line": 10, "body": "note"},  # empty path
        {"path": "x.py", "line": "ten", "body": "note"},  # unparseable
        {"path": "x.py", "line": None, "body": "note"},
        {"path": "x.py", "line": 10.5, "body": "note"},  # lossy
        {"path": "x.py", "line": True, "body": "note"},  # bool is an int
        "not an object",
    ],
)
def test_build_comments_drops_unpostable_findings(finding):
    comments, skipped = m.build_comments([finding], ANCHORS)
    assert comments == []
    assert len(skipped) == 1


def test_one_bad_finding_does_not_take_the_good_ones_with_it():
    """GitHub rejects the whole review for one bad position."""
    comments, skipped = m.build_comments(
        [
            {"path": "x.py", "line": 10, "body": "good"},
            {"path": "x.py", "line": 9999, "body": "bad anchor"},
            {"path": "x.py", "line": 11, "body": "also good"},
        ],
        ANCHORS,
    )
    assert [c["line"] for c in comments] == [10, 11]
    assert len(skipped) == 1


# --------------------------------------------------------------------------
# main() — end to end, as the workflow invokes it
# --------------------------------------------------------------------------


def _run(tmp_path: Path, response: str, diff: str) -> tuple[int, Path, str]:
    result = tmp_path / "agy_result.json"
    result.write_text(json.dumps({"response": response}), encoding="utf-8")
    diff_file = tmp_path / "pr_diff_used.txt"
    diff_file.write_text(diff, encoding="utf-8")
    out = tmp_path / "review_payload.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result",
            str(result),
            "--diff",
            str(diff_file),
            "--label",
            "Correctness",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, out, proc.stdout


DIFF = _diff("--- a/x.py", "+++ b/x.py", "@@ -1,1 +1,2 @@", " keep", "+added")


def test_main_writes_a_payload_the_rest_api_accepts(tmp_path):
    response = (
        '```json\n[{"path": "x.py", "line": 2, "body": "Off by one."}]\n```'
    )
    code, out, _ = _run(tmp_path, response, DIFF)
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["event"] == "COMMENT"
    assert payload["body"]  # required by the API for a COMMENT event
    assert payload["comments"] == [
        {"path": "x.py", "line": 2, "side": "RIGHT", "body": "Off by one."}
    ]


def test_main_writes_nothing_when_there_are_no_findings(tmp_path):
    code, out, _ = _run(tmp_path, "```json\n[]\n```", DIFF)
    assert code == 0
    assert not out.exists()


def test_main_writes_nothing_when_every_finding_is_dropped(tmp_path):
    response = '```json\n[{"path": "x.py", "line": 999, "body": "n"}]\n```'
    code, out, stdout = _run(tmp_path, response, DIFF)
    assert code == 0
    assert not out.exists()
    assert "dropped finding" in stdout
    # A dropped finding is not a contributor-facing annotation.
    assert "::warning::" not in stdout


def test_main_fails_loudly_on_unusable_reviewer_output(tmp_path):
    """A reviewer that returned nothing usable is a CI fault, not a verdict.

    It must not be reported as a problem with the pull request: exit with
    the dedicated CI-fault code and annotate this checker, never a file.
    """
    code, out, stdout = _run(tmp_path, "I found nothing.", DIFF)
    assert code == EXIT_CI_FAULT
    assert not out.exists()
    assert "[CI FAULT]" in stdout
    assert "post_review_comments.py" in stdout
    assert "file=" not in stdout  # never point at contributor code


def test_main_fails_when_the_result_file_is_not_json(tmp_path):
    result = tmp_path / "agy_result.json"
    result.write_text("", encoding="utf-8")
    diff_file = tmp_path / "d.txt"
    diff_file.write_text(DIFF, encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result",
            str(result),
            "--diff",
            str(diff_file),
            "--label",
            "Security",
            "--out",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == EXIT_CI_FAULT
    assert "[CI FAULT]" in proc.stdout


def test_main_fails_when_the_result_is_not_an_object(tmp_path):
    result = tmp_path / "agy_result.json"
    result.write_text("[1, 2, 3]", encoding="utf-8")
    diff_file = tmp_path / "d.txt"
    diff_file.write_text(DIFF, encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result",
            str(result),
            "--diff",
            str(diff_file),
            "--label",
            "Security",
            "--out",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == EXIT_CI_FAULT
    assert "not a JSON object" in proc.stdout


def test_diff_trimmed_mid_utf8_character_does_not_crash(tmp_path):
    """The workflow trims the diff with `head -c`, which cuts bytes.

    Sooner or later that cut lands inside a multi-byte character — any diff
    touching an em dash or an accent is a candidate. Strict decoding raised
    an uncaught UnicodeDecodeError and threw away the whole review.
    """
    whole = (
        "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,2 @@\n keep\n+caf\u00e9 au lait\n"
    )
    raw = whole.encode("utf-8")
    cut = raw[: raw.index("\u00e9".encode()) + 1]  # split the 2-byte é
    assert cut.decode("utf-8", "ignore") != cut.decode("utf-8", "replace")

    result = tmp_path / "agy_result.json"
    result.write_text(
        json.dumps(
            {
                "response": (
                    '```json\n[{"path": "x.py", "line": 2, '
                    '"body": "note"}]\n```'
                )
            }
        ),
        encoding="utf-8",
    )
    diff_file = tmp_path / "pr_diff_used.txt"
    diff_file.write_bytes(cut)
    out = tmp_path / "review_payload.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result",
            str(result),
            "--diff",
            str(diff_file),
            "--label",
            "Correctness",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr
    # The partial line is still an added line, so the finding survives.
    assert (
        json.loads(out.read_text(encoding="utf-8"))["comments"][0]["line"] == 2
    )


def test_workflow_invokes_this_script_with_the_flags_it_defines():
    """Pin the workflow -> CLI contract.

    post_review_comments.py is called from a shell block in
    _ai-pr-review-core.yml, so a renamed flag or a moved file is invisible to
    both ruff and pytest and only shows up as a failed review on a real PR.
    """
    # A hard import, not importorskip: pyyaml is a project dependency, so a
    # skip here could only ever mean this pin quietly stopped being enforced.
    import yaml

    workflow = (
        Path(__file__).resolve().parents[3]
        / ".github"
        / "workflows"
        / "_ai-pr-review-core.yml"
    )
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"][
        "review-pr"
    ]["steps"]
    post = next(s for s in steps if s.get("id") == "post_review")

    invocation = post["run"]
    assert "python3 .github/scripts/post_review_comments.py" in invocation
    for flag in ("--result", "--diff", "--label", "--out"):
        assert flag in invocation, f"workflow no longer passes {flag}"

    # ...and the script still defines exactly those flags.
    defined = {
        opt
        for action in m.build_parser()._actions
        for opt in action.option_strings
    }
    assert {"--result", "--diff", "--label", "--out"} <= defined
