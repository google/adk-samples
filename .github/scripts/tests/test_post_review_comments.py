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
import time
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
    comments, _notes, skipped = m.build_comments(
        [{"path": "x.py", "line": 10, "body": "Off by one."}], ANCHORS
    )
    assert comments == [
        {"path": "x.py", "line": 10, "side": "RIGHT", "body": "Off by one."}
    ]
    assert skipped == []


def test_build_comments_strips_a_stray_b_prefix():
    comments, _notes, _skipped = m.build_comments(
        [{"path": "b/x.py", "line": 10, "body": "note"}], ANCHORS
    )
    assert comments[0]["path"] == "x.py"


def test_a_real_path_starting_with_b_slash_is_not_stripped():
    """A repo with a top-level `b/` directory has real paths like this.

    Stripping unconditionally would turn `b/pkg.py` into `pkg.py`, match
    nothing, and silently drop every finding in that directory.
    """
    anchors = {"b/pkg.py": {7}}
    comments, _notes, skipped = m.build_comments(
        [{"path": "b/pkg.py", "line": 7, "body": "note"}], anchors
    )
    assert comments[0]["path"] == "b/pkg.py"
    assert skipped == []


def test_build_comments_accepts_a_stringified_line():
    comments, _notes, _skipped = m.build_comments(
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
    comments, _notes, skipped = m.build_comments([finding], ANCHORS)
    assert comments == []
    assert len(skipped) == 1


def test_one_bad_finding_does_not_take_the_good_ones_with_it():
    """GitHub rejects the whole review for one bad position."""
    comments, _notes, skipped = m.build_comments(
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
# implausible_body — the shape limits on what reaches a public PR
#
# The threat these answer is b/555419958: a body is model text derived from a
# diff a fork author wrote, and it is posted verbatim through the API, where
# log masking does not reach. The limits are not the boundary — the empty agy
# tool allowlist in the workflow is — so what these tests pin is the
# calibration, which is the part that rots. Both directions matter equally:
# too loose and the channel carries a credential, too tight and it silently
# drops real reviews.
# --------------------------------------------------------------------------

# The longest path in this repository at the time the cap was set. Quoting a
# path is what a legitimate finding does, so this is the shape
# MAX_UNBROKEN_RUN has to keep accepting. If a longer path ever lands, this
# fails here rather than by dropping a contributor's review comment.
LONGEST_REPO_PATH = (
    "java/agents/time-series-forecasting/src/main/java/com/google/adk/"
    "samples/agents/timeseriesforecasting/ForecastingAgent.java"
)


@pytest.mark.parametrize(
    "body",
    [
        "Off by one.",
        "This drops the error instead of raising it; re-raise after logging.",
        f"`{LONGEST_REPO_PATH}` is imported here but never used.",
        "See https://github.com/google/adk-samples/blob/main/AGENTS.md — "
        "gemini-2.5-flash is deprecated in this repo.",
        # Both caps exactly at their limit, which is where an off-by-one in
        # either comparison would show up.
        ("word " * m.MAX_BODY_CHARS)[: m.MAX_BODY_CHARS],
        "y" * m.MAX_UNBROKEN_RUN,
    ],
)
def test_a_real_review_comment_is_plausible(body):
    assert m.implausible_body(body) is None


def test_the_longest_repo_path_clears_the_run_cap():
    """Pins the calibration itself, not just its effect."""
    assert len(LONGEST_REPO_PATH) < m.MAX_UNBROKEN_RUN


def test_an_over_long_body_is_implausible():
    reason = m.implausible_body("x" * (m.MAX_BODY_CHARS + 1))
    assert reason is not None
    assert str(m.MAX_BODY_CHARS) in reason


def test_an_unbroken_run_past_the_cap_is_implausible():
    reason = m.implausible_body(
        "The value is " + "A" * (m.MAX_UNBROKEN_RUN + 1)
    )
    assert reason is not None
    assert "unbroken run" in reason


def test_a_credential_shaped_payload_is_implausible():
    """The concrete thing this exists to refuse.

    Shaped like the ADC file the runner holds: an external_account config
    whose credential_source embeds the runner's OIDC request token.
    """
    payload = json.dumps(
        {
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/123456789/locations/"
            "global/workloadIdentityPools/gh-pool/providers/gh-provider",
            "token_url": "https://sts.googleapis.com/v1/token",
            "credential_source": {
                "url": "https://pipelinesghubeus.actions.githubusercontent.com"
                "/abcdef/_apis/distributedtask/hubs/Actions/plans/0000/jobs"
                "/idtoken",
                "headers": {"Authorization": "bearer " + "e" * 400},
            },
        }
    )
    assert m.implausible_body(payload) is not None


def test_an_implausible_body_is_dropped_before_it_can_be_posted():
    comments, notes, skipped = m.build_comments(
        [{"path": "x.py", "line": 10, "body": "z" * (m.MAX_BODY_CHARS + 1)}],
        ANCHORS,
    )
    assert (comments, notes) == ([], [])
    assert len(skipped) == 1


def test_the_note_path_is_shape_checked_too():
    """A note is not posted inline, but it does reach the review body.

    Both are public, so a check covering only inline comments would leave half
    the channel open — and the note path is the easier half to reach, since it
    takes any window-verified line rather than only lines the PR adds.
    """
    diff = _diff(
        "--- a/x.py",
        "+++ b/x.py",
        "@@ -1,2 +1,3 @@",
        " def handler(req):",
        "-    return None",
        "+    return req",
    )
    anchors, text = m.walk_right_side(diff)
    comments, notes, skipped = m.build_comments(
        [
            {
                "path": "x.py",
                "line": 1,
                "body": "A" * (m.MAX_UNBROKEN_RUN + 1),
                "window": "  1: def handler(req):",
            }
        ],
        anchors,
        text,
    )
    assert (comments, notes) == ([], [])
    assert len(skipped) == 1


def test_one_implausible_body_does_not_take_the_good_ones_with_it():
    comments, _notes, skipped = m.build_comments(
        [
            {"path": "x.py", "line": 10, "body": "good"},
            {"path": "x.py", "line": 11, "body": "q" * (m.MAX_BODY_CHARS + 1)},
        ],
        ANCHORS,
    )
    assert [c["line"] for c in comments] == [10]
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
    # The `review` job, not `post`: building the payload and posting it are
    # deliberately different jobs, and this script runs in the one that holds
    # no write token. If that ever moves back into `post`, this fails — which
    # is the point, because the split is a security boundary (b/555419958).
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"][
        "review"
    ]["steps"]
    build = next(s for s in steps if s.get("id") == "build_review")

    invocation = build["run"]
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


# --------------------------------------------------------------------------
# Window verification
#
# The prompt's severity gate ("only critical or high") used to be the only
# thing between a wrong finding and a contributor's PR. It has been replaced
# by a wider one, so these checks are what keeps precision up. Each test below
# pins one of them.
# --------------------------------------------------------------------------

WINDOW_DIFF = _diff(
    "--- a/x.py",
    "+++ b/x.py",
    "@@ -1,2 +1,4 @@",
    " def handler(req):",
    "-    return None",
    "+    name = req.args['n']",
    "+    os.system(f'echo {name}')",
    "+    return name",
)


def _one(finding, diff=WINDOW_DIFF, existing=None):
    anchors, text = m.walk_right_side(diff)
    return m.build_comments([finding], anchors, text, existing)


def test_a_matching_window_is_kept():
    comments, notes, skipped = _one(
        {
            "path": "x.py",
            "line": 3,
            "body": "shell injection here",
            "window": "  2:     name = req.args['n']\n  3:     os.system(f'echo {name}')",
        }
    )
    assert [c["line"] for c in comments] == [3]
    assert (notes, skipped) == ([], [])


def test_a_fabricated_window_is_dropped():
    """A lane that invents a finding invents the source under it too.

    This is the check that makes a wider filter safe: without it the only
    defence against a hallucinated finding is a real line number, which a
    model guesses correctly often enough to be no defence at all.
    """
    comments, _notes, skipped = _one(
        {
            "path": "x.py",
            "line": 3,
            "body": "sql injection here",
            "window": "  3:     cursor.execute('SELECT ' + name)",
        }
    )
    assert comments == []
    assert "window says" in skipped[0]


def test_a_window_off_by_one_corrects_the_anchor():
    """Real finding, wrong arithmetic — repair it rather than lose it.

    Counting new-file line numbers out of a unified diff by hand is the part
    of the job a model is worst at, and dropping those findings throws away
    correct work over an off-by-one.
    """
    comments, _notes, skipped = _one(
        {
            "path": "x.py",
            "line": 4,
            "body": "shell injection here",
            "window": "  3:     name = req.args['n']\n  4:     os.system(f'echo {name}')",
        }
    )
    assert [c["line"] for c in comments] == [3]
    assert skipped == []


def test_a_finding_with_no_window_still_needs_a_real_added_line():
    """No window means no verification, so the old rule stands unchanged."""
    good, _n1, _s1 = _one({"path": "x.py", "line": 3, "body": "note"})
    bad, _n2, skipped = _one({"path": "x.py", "line": 99, "body": "note"})
    assert len(good) == 1
    assert bad == []
    assert "not a line this PR adds" in skipped[0]


def test_a_verified_finding_on_an_unchanged_line_becomes_a_note():
    """Real, but GitHub will not take an inline comment there.

    These used to go to the job log and vanish. On PR #2373 that class held
    all three of the hard CI failures the review found.
    """
    comments, notes, skipped = _one(
        {
            "path": "x.py",
            "line": 1,
            "body": "no type hints on this signature",
            "window": "  1: def handler(req):",
        }
    )
    assert comments == []
    assert notes == [
        {"path": "x.py", "line": 1, "body": "no type hints on this signature"}
    ]
    assert skipped == []


def test_an_unverified_finding_on_an_unchanged_line_is_not_promoted():
    """An unverifiable line number is model arithmetic, not a finding.

    Promoting those to the review body would surface exactly the mistakes
    window verification exists to catch.
    """
    comments, notes, skipped = _one(
        {
            "path": "x.py",
            "line": 1,
            "body": "note",
            "window": "  1: something that is not in this diff at all",
        }
    )
    assert (comments, notes) == ([], [])
    assert len(skipped) == 1


# --------------------------------------------------------------------------
# The cheapness gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "steps",
    [
        "trace the value back to its caller",
        "assuming the input is user-controlled, read line 3",
        "consider the case where the list is empty",
        "grep the repo for other callers",
    ],
)
def test_a_finding_that_admits_it_is_expensive_is_dropped(steps):
    """Cost to check, not severity, is what a comment is filtered on.

    Cheap and wrong costs the author five seconds; expensive and wrong costs
    twenty minutes and the credibility of every other comment in the review.
    """
    comments, _notes, skipped = _one(
        {
            "path": "x.py",
            "line": 3,
            "body": "note",
            "window": "  3:     os.system(f'echo {name}')",
            "verify_steps": steps,
        }
    )
    assert comments == []
    assert "not cheap to verify" in skipped[0]


def test_a_finding_settled_at_the_anchor_survives_the_gate():
    comments, _notes, skipped = _one(
        {
            "path": "x.py",
            "line": 3,
            "body": "note",
            "window": "  3:     os.system(f'echo {name}')",
            "verify_steps": "read line 3 of this file",
        }
    )
    assert len(comments) == 1
    assert skipped == []


# --------------------------------------------------------------------------
# Duplicate suppression
#
# Four lanes review every push. Without this each one repeats itself on every
# `synchronize` and repeats whatever the other three found in the overlap
# between their remits.
# --------------------------------------------------------------------------


def _existing(path, line, body):
    return [
        {
            "kind": "inline",
            "path": path,
            "line": line,
            "original_line": line,
            "body": body,
        }
    ]


def test_a_comment_already_on_the_line_suppresses_the_finding():
    comments, _notes, skipped = _one(
        {"path": "x.py", "line": 3, "body": "shell injection here"},
        existing=_existing("x.py", 3, "anything at all"),
    )
    assert comments == []
    assert "already commented on this line" in skipped[0]


def test_suppression_reaches_two_lines_either_side():
    comments, _notes, _skipped = _one(
        {"path": "x.py", "line": 4, "body": "note"},
        existing=_existing("x.py", 2, "something"),
    )
    assert comments == []


def test_suppression_does_not_reach_three_lines_away():
    """Tight on purpose — widening it starts eating genuinely new findings."""
    comments, _notes, _skipped = _one(
        {"path": "x.py", "line": 3, "body": "note"},
        existing=_existing("x.py", 6, "something"),
    )
    assert len(comments) == 1


def test_a_similar_comment_elsewhere_suppresses_the_finding():
    """The other three lanes phrase the same defect differently."""
    comments, _notes, skipped = _one(
        {
            "path": "x.py",
            "line": 3,
            "body": "unsanitised filename interpolated into os.system",
        },
        existing=_existing(
            "other.py",
            99,
            "filename is interpolated into os.system unsanitised",
        ),
    )
    assert comments == []
    assert "very similar" in skipped[0]


def test_an_unrelated_existing_comment_does_not_suppress():
    comments, _notes, _skipped = _one(
        {"path": "x.py", "line": 3, "body": "shell injection in this handler"},
        existing=_existing("x.py", 40, "please rename this fixture"),
    )
    assert len(comments) == 1


# --------------------------------------------------------------------------
# Output parsing and payload shape
# --------------------------------------------------------------------------


def test_the_last_fenced_block_wins():
    """The reviewer now reasons in prose before answering.

    That scan quotes diff rows and sometimes fences them, so the first block
    in the response is no longer reliably the answer. The prompt puts the
    findings last; this reads them from the same end.
    """
    response = (
        "Working through the diff.\n\n"
        '```json\n[{"path": "nope.py", "line": 1, "body": "an example"}]\n```\n\n'
        "Now the real answer.\n\n"
        '```json\n[{"path": "x.py", "line": 2, "body": "the real one"}]\n```\n'
    )
    assert m.extract_findings(response) == [
        {"path": "x.py", "line": 2, "body": "the real one"}
    ]


def test_notes_are_listed_in_the_review_body():
    payload = m.build_payload(
        "Correctness",
        [{"path": "x.py", "line": 2, "side": "RIGHT", "body": "inline"}],
        [{"path": "y.toml", "line": 9, "body": "requires-python is 3.10"}],
    )
    assert "`y.toml:9` — requires-python is 3.10" in payload["body"]
    assert len(payload["comments"]) == 1


def test_a_review_of_notes_alone_is_still_posted(tmp_path):
    """Nothing inline to say does not mean nothing to say."""
    response = json.dumps(
        [
            {
                "path": "x.py",
                "line": 1,
                "body": "no type hints here",
                "window": "  1: def handler(req):",
            }
        ]
    )
    code, out, _ = _run(tmp_path, f"```json\n{response}\n```", WINDOW_DIFF)
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["comments"] == []
    assert "no type hints here" in payload["body"]


def test_an_anchor_outside_its_own_window_is_pulled_into_it():
    """The window is checked against the diff; the anchor is not.

    A model that quotes the right code and then names the line underneath it
    would otherwise get a comment about a placeholder value posted on the
    `return` statement below it — right finding, wrong line, reads as
    carelessness.
    """
    comments, _notes, skipped = _one(
        {
            "path": "x.py",
            "line": 4,
            "body": "shell injection here",
            "window": "  3:     os.system(f'echo {name}')",
        }
    )
    assert [c["line"] for c in comments] == [3]
    assert skipped == []


def test_an_anchor_inside_its_window_is_left_alone():
    comments, _notes, _skipped = _one(
        {
            "path": "x.py",
            "line": 2,
            "body": "note",
            "window": "  2:     name = req.args['n']\n  3:     os.system(f'echo {name}')",
        }
    )
    assert [c["line"] for c in comments] == [2]


# --------------------------------------------------------------------------
# Salvaging a malformed findings block
#
# Findings quote source verbatim in `window`, and source is full of double
# quotes. A reviewer that forgets to escape one used to cost the whole review
# AND turn the check red. Caught by a dry run against a real PR, not by any
# test written beforehand — hence the recorded fixture.
# --------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


def test_one_unescaped_quote_does_not_destroy_the_whole_review():
    """Regression: run 32896537749, Maintainability on PR #2545.

    The reviewer emitted `"window": "  211:     assert res["success"] is True"`
    — valid but for one unescaped pair. Three good findings were thrown away
    and the job failed with a CI-fault annotation on the contributor's PR.

    All three come back now. Salvage used to keep the two findings that were
    already well-formed and drop the third; repair rebuilds the block, so the
    malformed one is no longer a casualty either — and its window keeps the
    quotes that broke it.
    """
    response = (FIXTURES / "malformed_findings_response.txt").read_text(
        encoding="utf-8"
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(m.FENCED_BLOCK.findall(response)[-1])

    findings = m.extract_findings(response)
    assert len(findings) == 3
    paths = {f["path"].split("/")[-1] for f in findings}
    assert paths == {"test_routine_tool.py", "test_process_tool.py"}

    recovered = next(f for f in findings if f["line"] == 212)
    assert 'assert res["success"] is True' in recovered["window"]


def test_salvage_keeps_only_things_shaped_like_findings():
    """Advancing past a bad object can land on a `{` inside a string.

    Requiring path and body is what stops that debris becoming a comment.
    """
    block = (
        '[{"path": "a.py", "line": 1, "body": "real", "window": "x"broken"},'
        ' {"nested": {"not": "a finding"}},'
        ' {"path": "b.py", "line": 2, "body": "also real"}]'
    )
    salvaged = m._salvage_findings(block, json.JSONDecoder())
    assert [f["path"] for f in salvaged] == ["b.py"]


def test_a_block_that_salvages_nothing_is_still_a_ci_fault():
    """Silence must not be mistaken for a clean review.

    Nothing here carries a finding key, so there is no value for repair to
    delimit and nothing for salvage to keep. The fault must survive both.
    """
    with pytest.raises(m.ReviewerOutputError):
        m.extract_findings("```json\n[{totally broken}\n```")


def test_every_finding_malformed_still_yields_a_review():
    """Regression: the Correctness lane on PR #2566.

    Both findings quoted shell out of a workflow file, so both windows came
    back with raw quotes and salvage — which only keeps findings that were
    already well-formed — recovered nothing. The review died as a CI fault
    over output that was one escape away from usable.
    """
    response = (FIXTURES / "unescaped_quotes_response.txt").read_text(
        encoding="utf-8"
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(m.FENCED_BLOCK.findall(response)[-1])
    assert (
        m._salvage_findings(
            m.FENCED_BLOCK.findall(response)[-1], json.JSONDecoder()
        )
        == []
    )

    findings = m.extract_findings(response)
    assert len(findings) == 2
    assert all(
        f["path"] == ".github/workflows/typescript-tests.yml" for f in findings
    )
    assert '[ -f "yarn.lock" ]' in findings[0]["window"]
    assert '[ -f "bun.lockb" ]' in findings[1]["window"]


def test_a_bracket_after_a_stray_quote_is_not_a_value_end():
    """The trap that rules out the cheap way of finding a value's end.

    `[ -f "x.lock" ]` puts a `]` immediately after the stray quote. Ending a
    value at the next quote followed by `,`, `]` or `}` would stop there,
    truncating the array mid-string and losing every finding after it. Only a
    schema boundary — the next key, or the close of the object — ends a value.
    """
    block = '{"window": "  1: [ -f "x.lock" ]; then", "verify_steps": "x"}'
    start = block.index('"  1:') + 1

    end = m.STRING_FIELD_END.search(block, start)
    assert end is not None
    assert block[start : end.start()] == '  1: [ -f "x.lock" ]; then'


def test_repair_leaves_a_field_it_cannot_delimit_alone():
    """An unterminated value has no schema anchor after it.

    Rewriting on a guess would corrupt the one field still readable, so there
    is no reading to accept and the existing paths decide.
    """
    assert (
        m._repaired_findings('[{"path": "a.py", "body": "unterminated') is None
    )


def test_escaping_a_value_is_idempotent():
    """A value already escaped must not gain a second layer of backslashes.

    Repair normalises before it escapes, so re-reading a block that was only
    partly malformed cannot double up the quotes that were already correct.
    """
    assert m._escape_value('says \\"hi\\" loudly') == 'says \\"hi\\" loudly'
    assert m._escape_value('says "hi" loudly') == 'says \\"hi\\" loudly'


def test_repair_refuses_a_window_that_quotes_findings_shaped_source():
    """A second reading scrapes the finding's own fields out of the window.

    The window quotes source that is itself a findings array — which this
    repo's own tests contain — so one reading anchors at line 919 with the
    body "dup", and another stops the value early and takes `line` 1 and the
    body "real" from inside the quoted source. Both parse, so parsing cannot
    be the test. Posting the second would put source text on a contributor's
    PR as though it were a review, so the block is declined outright.
    """
    block = (
        '[{"path": "t.py", "line": 919, "body": "dup",'
        ' "window": "  919: [{"path": "a.py", "line": 1, "body": "real"}]"}]'
    )
    assert m._repaired_findings(block) is None


def test_repair_declines_rather_than_guessing_past_its_budget():
    """An exhausted search has not established that a reading is unique."""
    block = (
        '[{"path": "a.py", "line": 1, "body": "b",'
        ' "window": "' + '", "body": "x' * 40 + '"}]'
    )
    assert m._repaired_findings(block) is None


def _wide_malformed_block(findings: int) -> str:
    """A block with far more findings than a review ever asks for, one bad."""
    body = ",".join(
        f'{{"path": "f{i}.py", "line": 1, "body": "b", "window": "  1: x"}}'
        for i in range(findings)
    )
    return f"[{body}]".replace("  1: x", '  1: [ -f "a.lock" ]', 1)


def test_a_runaway_block_does_not_blow_the_stack():
    """Recursion made a wide block fatal instead of merely unreadable.

    One field was one frame, so a model answering with hundreds of findings
    raised RecursionError — and that escapes extract_findings entirely, so a
    block salvage could have read reached the CI fault instead. The walk is
    iterative for that reason; here it must simply decline.
    """
    assert m._repaired_findings(_wide_malformed_block(400)) is None


def test_a_runaway_block_is_bounded_in_time():
    """Charging only completed readings let a wide block run for minutes.

    Every later field boundary is a candidate end for every earlier field, so
    a 400-finding block pushed ~1600 copies of the reading per step and took
    over two minutes. Pushes are charged to the budget, and the ends weighed
    per field are capped, so the walk is bounded by shape as well as depth.
    """
    start = time.monotonic()
    assert m._repaired_findings(_wide_malformed_block(1000)) is None
    assert time.monotonic() - start < 5


def test_a_runaway_block_still_reaches_salvage():
    """Declining to repair must hand the block on, not end the review.

    The findings that were well-formed are still there to be kept, and before
    the walk was bounded they were lost with the rest.
    """
    findings = m.extract_findings(f"```json\n{_wide_malformed_block(400)}\n```")
    assert len(findings) == 399
    assert all(f["path"].endswith(".py") for f in findings)
