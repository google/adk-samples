"""The deterministic review lane: recipe discovery, translation, isolation."""

import json
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
