#!/usr/bin/env python3
"""Unit tests for tools/validate_manifest.py.

The `validate` / `validate_manifest` modules are importable directly because
the repo root is installed as the `adk-samples-tools` package (see the hatch
`sources` mapping in pyproject.toml), so no sys.path shim is needed here.
"""

import textwrap
from pathlib import Path

import pytest
import validate_manifest as m

VALID_MANIFEST = textwrap.dedent(
    """\
    type: standalone
    status: active
    language: python
    description: A valid recipe description that is comfortably long.
    ownership:
      team: My Team
      poc: my-github-id
    """
)


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_recipe(
    root: Path, rel: str, manifest: str | None = VALID_MANIFEST
) -> Path:
    """Create a recipe dir at root/rel. If manifest is None, omit manifest.yaml
    but still add a file so the dir qualifies as a recipe."""
    recipe = root / rel
    recipe.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        _write(recipe / "agent.py", "# placeholder\n")
    else:
        _write(recipe / "manifest.yaml", manifest)
    return recipe


# ---------------------------------------------------------------------------
# is_recipe_dir
# ---------------------------------------------------------------------------


def test_is_recipe_dir_true_with_content(tmp_path):
    d = tmp_path / "my-recipe"
    _write(d / "manifest.yaml", VALID_MANIFEST)
    assert m.is_recipe_dir(d) is True


def test_is_recipe_dir_false_readme_only(tmp_path):
    d = tmp_path / "docs-only"
    _write(d / "README.md", "# only readme")
    assert m.is_recipe_dir(d) is False


def test_is_recipe_dir_false_for_language_namespace(tmp_path):
    d = tmp_path / "python"
    _write(d / "some-recipe" / "manifest.yaml", VALID_MANIFEST)
    assert m.is_recipe_dir(d) is False


def test_is_recipe_dir_false_for_hidden_dir(tmp_path):
    d = tmp_path / ".hidden"
    _write(d / "manifest.yaml", VALID_MANIFEST)
    assert m.is_recipe_dir(d) is False


def test_is_recipe_dir_false_for_file(tmp_path):
    f = _write(tmp_path / "afile.txt", "x")
    assert m.is_recipe_dir(f) is False


# ---------------------------------------------------------------------------
# validate_manifest (uses the real schema)
# ---------------------------------------------------------------------------


def test_validate_manifest_valid(tmp_path):
    schema = m.load_schema()
    manifest = _write(tmp_path / "manifest.yaml", VALID_MANIFEST)
    assert m.validate_manifest(manifest, schema) == []


def test_validate_manifest_empty_file(tmp_path):
    schema = m.load_schema()
    manifest = _write(tmp_path / "manifest.yaml", "")
    errors = m.validate_manifest(manifest, schema)
    assert errors == ["manifest.yaml is empty"]


def test_validate_manifest_bad_yaml(tmp_path):
    schema = m.load_schema()
    manifest = _write(tmp_path / "manifest.yaml", "type: [unclosed\n")
    errors = m.validate_manifest(manifest, schema)
    assert any("YAML parse error" in e for e in errors)


def test_validate_manifest_missing_required_field(tmp_path):
    schema = m.load_schema()
    # Missing 'ownership'
    content = textwrap.dedent(
        """\
        type: standalone
        status: active
        language: python
        description: A valid recipe description that is long enough.
        """
    )
    manifest = _write(tmp_path / "manifest.yaml", content)
    errors = m.validate_manifest(manifest, schema)
    assert errors
    assert any("ownership" in e for e in errors)


def test_validate_manifest_placeholder_team_and_poc(tmp_path):
    schema = m.load_schema()
    content = textwrap.dedent(
        f"""\
        type: standalone
        status: active
        language: python
        description: A valid recipe description that is long enough.
        ownership:
          team: "{m.OWNERSHIP_TEAM_PLACEHOLDER}"
          poc: "{m.OWNERSHIP_POC_PLACEHOLDER}"
        """
    )
    manifest = _write(tmp_path / "manifest.yaml", content)
    errors = m.validate_manifest(manifest, schema)
    assert any("ownership.team" in e for e in errors)
    assert any("ownership.poc" in e for e in errors)


def test_validate_manifest_placeholder_description(tmp_path):
    # A TODO placeholder description is long enough to pass the schema's
    # minLength, so it must be caught by the explicit placeholder guard.
    schema = m.load_schema()
    content = textwrap.dedent(
        """\
        type: standalone
        status: active
        language: python
        description: "TODO: Replace with a clear description of what this recipe demonstrates (min 10 characters)."
        ownership:
          team: My Team
          poc: my-github-id
        """
    )
    manifest = _write(tmp_path / "manifest.yaml", content)
    errors = m.validate_manifest(manifest, schema)
    assert any("description" in e for e in errors)


# ---------------------------------------------------------------------------
# collect_recipe_dirs (monkeypatch REPO_ROOT to a fake tree)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A fake repo with flat and namespaced recipes under core/ and contrib/."""
    _make_recipe(tmp_path, "core/recipe-a")
    _make_recipe(tmp_path, "core/python/recipe-b")
    _make_recipe(tmp_path, "contrib/recipe-c")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    return tmp_path


def _rel(paths, root):
    return {str(p.relative_to(root)) for p in paths}


def test_collect_all(fake_repo):
    dirs = m.collect_recipe_dirs(None)
    assert _rel(dirs, fake_repo) == {
        "core/recipe-a",
        "core/python/recipe-b",
        "contrib/recipe-c",
    }


def test_collect_single_root(fake_repo):
    dirs = m.collect_recipe_dirs("core")
    assert _rel(dirs, fake_repo) == {"core/recipe-a", "core/python/recipe-b"}


def test_collect_language_namespace(fake_repo):
    dirs = m.collect_recipe_dirs("core/python")
    assert _rel(dirs, fake_repo) == {"core/python/recipe-b"}


def test_collect_single_flat_recipe(fake_repo):
    dirs = m.collect_recipe_dirs("core/recipe-a")
    assert _rel(dirs, fake_repo) == {"core/recipe-a"}


def test_collect_single_namespaced_recipe(fake_repo):
    dirs = m.collect_recipe_dirs("core/python/recipe-b")
    assert _rel(dirs, fake_repo) == {"core/python/recipe-b"}


def test_collect_nonexistent_scope_exits(fake_repo):
    with pytest.raises(SystemExit):
        m.collect_recipe_dirs("core/does-not-exist")


def test_collect_invalid_recipe_dir_exits(tmp_path, monkeypatch):
    # A dir with only README.md is not a valid recipe dir.
    readme_only = tmp_path / "core" / "readme-only"
    _write(readme_only / "README.md", "# just docs")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit):
        m.collect_recipe_dirs("core/readme-only")


# ---------------------------------------------------------------------------
# main (end-to-end against a fake tree)
# ---------------------------------------------------------------------------


def test_main_all_valid_returns_zero(tmp_path, monkeypatch):
    _make_recipe(tmp_path, "core/good-a")
    _make_recipe(tmp_path, "core/good-b")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    assert m.main("core") == 0


def test_main_missing_manifest_returns_one(tmp_path, monkeypatch):
    _make_recipe(tmp_path, "core/no-manifest", manifest=None)
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    assert m.main("core") == 1


def test_main_invalid_manifest_returns_one(tmp_path, monkeypatch):
    _make_recipe(tmp_path, "core/bad", manifest="type: not_a_valid_enum\n")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    assert m.main("core/bad") == 1


def test_main_missing_manifest_emits_github_annotation(
    tmp_path, monkeypatch, capsys
):
    # CI relies on the ::error file=...:: annotation format for PR feedback.
    _make_recipe(tmp_path, "core/no-manifest", manifest=None)
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    assert m.main("core") == 1

    out = capsys.readouterr().out
    assert "::error file=core/no-manifest/manifest.yaml::" in out
    assert "is missing" in out


def test_main_invalid_manifest_emits_github_annotation(
    tmp_path, monkeypatch, capsys
):
    _make_recipe(tmp_path, "core/bad", manifest="type: not_a_valid_enum\n")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    assert m.main("core/bad") == 1

    out = capsys.readouterr().out
    assert "::error file=core/bad/manifest.yaml::" in out
