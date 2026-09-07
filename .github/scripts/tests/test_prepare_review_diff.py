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
"""Unit tests for prepare_review_diff.py.

This script decides what the AI reviewer is allowed to see and how many
comments it is asked for. Both failure modes are quiet: dropping a file the
reviewer should have read produces a green check on unreviewed code, and a
budget computed off the wrong churn produces a review that is too thin or too
noisy for the PR it is on.

Every test below pins one of those.
"""

import subprocess
import sys
from pathlib import Path

import post_review_comments
import prepare_review_diff as m
import pytest

SCRIPT = Path(m.__file__)


def _section(path: str, *rows: str) -> str:
    return "\n".join(
        [
            f"diff --git a/{path} b/{path}",
            "index 111..222 100644",
            f"--- a/{path}",
            f"+++ b/{path}",
            *rows,
        ]
    )


CODE = _section("pkg/agent.py", "@@ -1,1 +1,2 @@", " keep", "+added")


# --------------------------------------------------------------------------
# What gets dropped
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "uv.lock",
        "recipe/uv.lock",
        "package-lock.json",
        "vendor/dep/thing.go",
        "web/node_modules/pkg/index.js",
        "dist/bundle.js",
        "app/__snapshots__/view.snap",
        "static/logo.png",
        "proto/service_pb2.py",
        "tests/testdata/big.json",
    ],
)
def test_unreviewable_files_are_dropped(path):
    """Every one of these displaces hand-written code from the prompt.

    The prompt has a hard byte budget, so on a PR that mixes a regenerated
    lockfile with real code the lockfile wins by being longer and the code it
    was generated from never reaches the model.
    """
    diff = _section(path, "@@ -1,1 +1,2 @@", " keep", "+added") + "\n" + CODE
    filtered, stats = m.filter_diff(diff)
    assert path not in filtered
    assert "pkg/agent.py" in filtered
    assert stats["kept_files"] == 1


def test_a_small_yaml_file_is_kept():
    """Size is doing the work, not the extension.

    A 20-line manifest.yaml is hand-written and very much worth reviewing; a
    5000-line one is a data dump.
    """
    diff = _section(
        "recipe/manifest.yaml", "@@ -1,1 +1,2 @@", " keep", "+added"
    )
    filtered, stats = m.filter_diff(diff)
    assert "manifest.yaml" in filtered
    assert stats["kept_files"] == 1


def test_a_large_data_file_is_dropped_whatever_it_is_called():
    rows = ["@@ -1,1 +1,600 @@"] + [f"+row {n}" for n in range(600)]
    diff = _section("eval/cases.json", *rows)
    _filtered, stats = m.filter_diff(diff)
    assert stats["kept_files"] == 0
    assert stats["skipped"][0][1] == "bulk data"


def test_a_deleted_file_is_dropped():
    """Nothing to fix in code the PR removes, and no RIGHT side to anchor to."""
    diff = "\n".join(
        [
            "diff --git a/old.py b/old.py",
            "deleted file mode 100644",
            "--- a/old.py",
            "+++ /dev/null",
            "@@ -1,2 +0,0 @@",
            "-gone",
            "-also gone",
        ]
    )
    _filtered, stats = m.filter_diff(diff)
    assert stats["kept_files"] == 0
    assert stats["reviewable_lines"] == 0


def test_a_pure_rename_is_dropped():
    """A file moved with no content change has nothing to review.

    On a migration PR this is most of the diff — 65 of 123 files on #2373.
    """
    diff = "\n".join(
        [
            "diff --git a/a/x.py b/b/x.py",
            "similarity index 100%",
            "rename from a/x.py",
            "rename to b/x.py",
        ]
    )
    _filtered, stats = m.filter_diff(diff)
    assert stats["kept_files"] == 0
    assert stats["skipped"][0][1] == "no content change"


def test_an_unparseable_diff_is_passed_through_whole():
    """Degrade to reviewing everything, never to reviewing nothing.

    Silently discarding a diff we failed to parse would turn a format change
    into a green check on an unreviewed PR.
    """
    diff = "something that is not a git diff at all\n"
    filtered, stats = m.filter_diff(diff)
    assert "not a git diff" in filtered
    assert stats["kept_files"] == 0


# --------------------------------------------------------------------------
# Churn and budget
# --------------------------------------------------------------------------


def test_churn_counts_both_sides_but_not_the_file_headers():
    diff = _section(
        "pkg/agent.py", "@@ -1,2 +1,2 @@", " keep", "-removed", "+added"
    )
    _filtered, stats = m.filter_diff(diff)
    assert stats["reviewable_lines"] == 2


def test_a_lockfile_does_not_inflate_the_budget():
    """The whole reason the budget is computed after filtering.

    A PR that is 1400 lines of regenerated lockfile plus 80 lines of code is
    a small PR, and asking for a large-PR number of comments on it produces
    padding.
    """
    lock_rows = ["@@ -1,1 +1,1400 @@"] + [f"+dep {n}" for n in range(1400)]
    diff = _section("uv.lock", *lock_rows) + "\n" + CODE
    _filtered, stats = m.filter_diff(diff)
    assert stats["reviewable_lines"] == 1
    assert m.budget_for(stats["reviewable_lines"]) == 2


@pytest.mark.parametrize(
    ("churn", "expected"),
    [(0, 2), (49, 2), (50, 2), (51, 3), (200, 3), (201, 5), (100_000, 5)],
)
def test_budget_scales_with_reviewable_churn(churn, expected):
    assert m.budget_for(churn) == expected


def test_the_per_lane_budget_never_exceeds_the_global_cap():
    """Four lanes run concurrently and cannot coordinate.

    Each is given a share rather than the whole budget, so the total stays at
    the cap however many findings the other three come back with.
    """
    largest = max(budget for _ceiling, budget in m.BUDGET_TABLE)
    assert largest * m.LANE_COUNT <= m.GLOBAL_CAP


# --------------------------------------------------------------------------
# main() — as the workflow invokes it
# --------------------------------------------------------------------------


def _run(tmp_path: Path, diff: str) -> tuple[int, str, dict]:
    src = tmp_path / "pr_diff.txt"
    src.write_text(diff, encoding="utf-8")
    out = tmp_path / "pr_diff_reviewable.txt"
    gho = tmp_path / "github_output"
    gho.touch()
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--diff",
            str(src),
            "--out",
            str(out),
            "--github-output",
            str(gho),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    outputs = dict(
        row.split("=", 1)
        for row in gho.read_text(encoding="utf-8").splitlines()
        if "=" in row
    )
    return proc.returncode, out.read_text(encoding="utf-8"), outputs


def test_main_writes_the_filtered_diff_and_its_outputs(tmp_path):
    code, filtered, outputs = _run(tmp_path, CODE)
    assert code == 0
    assert "pkg/agent.py" in filtered
    assert outputs["reviewable"] == "true"
    assert outputs["reviewable_lines"] == "1"
    assert outputs["budget"] == "2"


def test_main_reports_a_lockfile_only_pr_as_unreviewable(tmp_path):
    """This is what stops the workflow burning a model call on nothing.

    It also has to be a clean skip rather than a failure: a Dependabot PR is
    not a broken PR.
    """
    lock = _section("uv.lock", "@@ -1,1 +1,2 @@", " keep", "+added")
    code, filtered, outputs = _run(tmp_path, lock)
    assert code == 0
    assert filtered.strip() == ""
    assert outputs["reviewable"] == "false"


def test_the_filtered_diff_still_parses_as_a_diff(tmp_path):
    """The reviewer's anchors are computed from whatever survives here.

    A filter that corrupted the diff structure would put comments on real but
    wrong lines, which passes validation and gets posted.
    """
    diff = (
        _section("uv.lock", "@@ -1,1 +1,2 @@", " keep", "+dep")
        + "\n"
        + _section("pkg/agent.py", "@@ -10,1 +10,2 @@", " keep", "+added")
    )
    _code, filtered, _outputs = _run(tmp_path, diff)
    assert post_review_comments.added_line_anchors(filtered) == {
        "pkg/agent.py": {11}
    }


def test_workflow_invokes_this_script_with_the_flags_it_defines():
    """Pin the workflow -> CLI contract.

    The script is called from a shell block in _ai-pr-review-core.yml, so a
    renamed flag or a moved file is invisible to both ruff and pytest and only
    shows up as a review that never runs.
    """
    import yaml

    workflow = (
        Path(__file__).resolve().parents[3]
        / ".github"
        / "workflows"
        / "_ai-pr-review-core.yml"
    )
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"][
        "review"
    ]["steps"]
    prepare = next(s for s in steps if s.get("id") == "prepare_diff")

    invocation = prepare["run"]
    assert "python3 .github/scripts/prepare_review_diff.py" in invocation
    for flag in ("--diff", "--out", "--github-output"):
        assert flag in invocation, f"workflow no longer passes {flag}"

    defined = {
        opt
        for action in m.build_parser()._actions
        for opt in action.option_strings
    }
    assert {"--diff", "--out", "--github-output"} <= defined


def test_the_budget_the_workflow_reads_is_the_one_this_script_writes():
    """The prompt interpolates ${{ steps.prepare_diff.outputs.budget }}.

    An output renamed here and not there yields an empty budget line in the
    prompt, which reads as "no budget" and puts the reviewer straight back to
    the one-or-two comments this change exists to fix.
    """
    core = (
        Path(__file__).resolve().parents[3]
        / ".github"
        / "workflows"
        / "_ai-pr-review-core.yml"
    ).read_text(encoding="utf-8")
    for name in ("budget", "reviewable_lines", "reviewable"):
        assert f"steps.prepare_diff.outputs.{name}" in core
