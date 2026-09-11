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

"""The deterministic review lane: recipe discovery, translation, isolation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import house_rules_lane as lane

CHECKER = (
    Path(__file__).resolve().parents[3]
    / ".agents"
    / "skills"
    / "github-pr-review"
    / "scripts"
    / "check_house_rules.py"
)


# --------------------------------------------------------- recipe discovery


def test_only_the_recipes_the_pr_touches_are_checked():
    """A PR editing one recipe must not collect findings about the other 400."""
    roots = lane.recipe_roots(
        [
            "contrib/python/my-recipe/agent.py",
            "contrib/python/my-recipe/pyproject.toml",
            "core/go/other/main.go",
            "skills/retail/store-ops/manifest.yaml",
            "README.md",
            ".github/workflows/thing.yml",
            "tools/validate_structure.py",
        ]
    )
    assert roots == [
        "contrib/python/my-recipe",
        "core/go/other",
        "skills/retail/store-ops",
    ]


def test_a_pr_touching_no_recipe_yields_nothing():
    assert lane.recipe_roots(["docs/x.md", ".github/policy.yml"]) == []


def test_a_recipe_root_is_three_segments_not_a_prefix():
    """`contrib/python` alone is not a recipe, and neither is the repo root."""
    assert lane.recipe_roots(["contrib/python/README.md"]) == []
    assert lane.recipe_roots(["skills/retail"]) == []


# ------------------------------------------------------------- translation


def test_the_citation_is_carried_into_the_comment_body():
    """A file the author can open is what stops a rule comment sounding
    arbitrary."""
    out = lane.to_reviewer_finding(
        {
            "path": "contrib/python/x/pyproject.toml",
            "line": 4,
            "what": "declares a [tool.ruff] table; recipes must not",
            "evidence": "python-validate-recipe.yml:261-266",
            "verify_steps": "grep '^\\[tool\\.ruff' in this file",
            "rule": "H1",
            "ci": "fail",
        }
    )
    assert out["body"].endswith("(python-validate-recipe.yml:261-266)")
    assert out["line"] == 4
    assert out["_rule"] == "H1"


def test_a_citation_already_in_the_body_is_not_repeated():
    out = lane.to_reviewer_finding(
        {
            "path": "p",
            "line": 1,
            "evidence": "AGENTS.md:40",
            "what": "deprecated model id, see AGENTS.md:40",
        }
    )
    assert out["body"].count("AGENTS.md:40") == 1


def test_no_window_is_emitted():
    """The window check audits a model's claim about the source. A checker that
    read the file makes no such claim, and supplying one only adds a way for
    this lane to fail."""
    assert "window" not in lane.to_reviewer_finding(
        {"path": "p", "line": 1, "what": "x"}
    )


def test_a_missing_line_anchors_at_the_top_of_the_file():
    assert lane.to_reviewer_finding({"path": "p", "what": "x"})["line"] == 1


# ------------------------------------------------------------- end to end


def _recipe(tmp_path: Path, rel: str, **files: str) -> Path:
    root = tmp_path / rel
    root.mkdir(parents=True)
    for name, text in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


MANIFEST = (
    "type: standalone\nstatus: active\nlanguage: python\n"
    'description: "A real description, long enough to pass."\n'
    'ownership:\n  team: "google"\n  poc: "someone"\n'
)


@pytest.mark.skipif(not CHECKER.exists(), reason="checker not in this checkout")
def test_findings_come_back_in_the_posting_scripts_shape(tmp_path):
    _recipe(
        tmp_path,
        "contrib/python/my-recipe",
        **{"manifest.yaml": MANIFEST, "pyproject.toml": "[tool.ruff]\n"},
    )
    changed = tmp_path / "changed.txt"
    changed.write_text("contrib/python/my-recipe/manifest.yaml\n")
    out = tmp_path / "findings.json"

    rc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "house_rules_lane.py"),
            "--checker",
            str(CHECKER),
            "--repo-root",
            str(tmp_path),
            "--changed-files",
            str(changed),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rc.returncode == 0, rc.stderr

    findings = json.loads(out.read_text())
    assert findings, "expected the junk ownership.team to be reported"
    for finding in findings:
        assert set(finding) >= {"path", "line", "body", "verify_steps"}
        assert isinstance(finding["line"], int)
        assert finding["path"].startswith("contrib/python/my-recipe/")
    assert any("google" in f["body"] for f in findings)


@pytest.mark.skipif(not CHECKER.exists(), reason="checker not in this checkout")
def test_state_does_not_leak_between_recipes(tmp_path):
    """Two recipes in one process. The checker keeps per-run state — the
    changed-file set, the skipped list, the git-tracked cache — and the second
    recipe must not inherit the first one's."""
    _recipe(tmp_path, "contrib/python/clean", **{"manifest.yaml": MANIFEST})
    _recipe(
        tmp_path,
        "contrib/python/dirty",
        **{"manifest.yaml": MANIFEST.replace('"google"', '"Real Team Name"')},
    )
    module = lane.load_checker(CHECKER)

    dirty, _ = lane.run_checker(
        module, str(tmp_path), "contrib/python/dirty", None
    )
    clean, _ = lane.run_checker(
        module, str(tmp_path), "contrib/python/clean", None
    )
    assert not [f for f in dirty if f["rule"] == "H48"]
    assert [f for f in clean if f["rule"] == "H48"], (
        "the second recipe's junk team was missed — state carried over"
    )
    assert all(f["path"].startswith("contrib/python/clean/") for f in clean)


@pytest.mark.skipif(not CHECKER.exists(), reason="checker not in this checkout")
def test_one_broken_recipe_does_not_cost_the_others(tmp_path, capsys):
    """A single unreadable recipe must not throw away every finding in the
    rest of the PR."""
    _recipe(tmp_path, "contrib/python/good", **{"manifest.yaml": MANIFEST})
    changed = tmp_path / "changed.txt"
    changed.write_text(
        "contrib/python/good/manifest.yaml\n"
        "contrib/python/vanished/manifest.yaml\n"
    )
    out = tmp_path / "findings.json"
    rc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "house_rules_lane.py"),
            "--checker",
            str(CHECKER),
            "--repo-root",
            str(tmp_path),
            "--changed-files",
            str(changed),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rc.returncode == 0, rc.stderr
    assert json.loads(out.read_text()), "the good recipe's findings were lost"


def test_ci_failures_are_ordered_first(tmp_path):
    """An author acts on "this blocks the build" and may never act on a nit."""
    findings = [
        {"path": "b", "line": 1, "what": "nit", "ci": "advisory"},
        {"path": "a", "line": 1, "what": "blocker", "ci": "fail"},
    ]
    translated = [lane.to_reviewer_finding(f) for f in findings]
    translated.sort(key=lambda f: (f.get("_ci") != "fail", f.get("path") or ""))
    assert [f["body"] for f in translated] == ["blocker", "nit"]


def test_h42_is_reported_once_for_the_whole_pr(tmp_path):
    """H42 is a property of the pull request, not of a recipe. Called inside
    the per-recipe loop it produced one identical comment per recipe on one
    line — and at exactly three recipes the grouping pass collapsed them into
    "the same thing in 2 other places", which is false: it is the same place,
    three times. The checker's own once-guard cannot see across recipes
    because `out` is fresh for each."""
    import inspect

    assert "module.check_pr_shape(" not in inspect.getsource(
        lane.run_checker
    ), "the per-recipe path still calls it, so it fires once per recipe"
    assert "module.check_pr_shape(" in inspect.getsource(lane.main), (
        "nothing calls it at all, so H42 never fires"
    )


@pytest.mark.skipif(not CHECKER.exists(), reason="checker not in this checkout")
def test_a_mistyped_project_table_does_not_delete_a_recipes_review(tmp_path):
    """`project = "oops"` is a realistic typo. .get() on a str raised out of
    the lane, which catches per recipe and reports the PR as clean."""
    _recipe(
        tmp_path,
        "contrib/python/typo",
        **{"manifest.yaml": MANIFEST, "pyproject.toml": 'project = "oops"\n'},
    )
    module = lane.load_checker(CHECKER)
    findings, _ = lane.run_checker(
        module, str(tmp_path), "contrib/python/typo", None
    )
    assert findings, "the recipe produced no findings at all"


@pytest.mark.skipif(not CHECKER.exists(), reason="checker not in this checkout")
def test_the_schema_comes_from_the_base_checkout(tmp_path):
    """A PR that edits its own manifest schema must not thereby edit the rule
    that judges it."""
    assert lane.SCHEMA_PATH.is_file()
    assert "pr-head" not in str(lane.SCHEMA_PATH)
    import inspect

    source = inspect.getsource(lane.run_checker)
    assert "SCHEMA_PATH" in source
    assert 'repo_root) / ".github/schemas' not in source


def test_an_all_recipes_failed_run_is_not_reported_as_clean(tmp_path, capsys):
    """The PR-shape check runs outside the per-recipe loop and contributes to
    the total, so one advisory nit from it made "every recipe failed" look
    like "we found something" and the lane exited green on a broken checker."""
    _recipe(tmp_path, "contrib/python/alpha", **{"manifest.yaml": MANIFEST})
    changed = tmp_path / "changed.txt"
    changed.write_text(
        "contrib/python/alpha/manifest.yaml\n.agents/skills/x/SKILL.md\n"
    )
    out = tmp_path / "findings.json"
    rc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "house_rules_lane.py"),
            "--checker",
            str(CHECKER),
            "--repo-root",
            str(tmp_path),
            "--changed-files",
            str(changed),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "tools"),
        },
    )
    # Sanity: the healthy case exits 0 and does find the mixed-PR nit.
    assert rc.returncode == 0, rc.stderr
    assert any(f["_rule"] == "H42" for f in json.loads(out.read_text()))
