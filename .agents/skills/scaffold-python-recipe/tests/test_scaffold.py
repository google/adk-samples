# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0
"""Unit tests for the scaffold-python-recipe skill script."""

import os
import shutil
from pathlib import Path

import pytest
import scaffold as m


def test_replace_in_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello <RECIPE_NAME> in <OUTPUT_DIRECTORY>", encoding="utf-8")

    m.replace_in_file(
        str(f),
        {"<RECIPE_NAME>": "my-recipe", "<OUTPUT_DIRECTORY>": "contrib"},
    )

    assert f.read_text(encoding="utf-8") == "Hello my-recipe in contrib"


def test_scaffold_creates_tree_and_replaces_placeholders(tmp_path):
    name = "my-test-recipe"

    ok = m.scaffold(name=name, output_dir=str(tmp_path))

    assert ok is True
    target = tmp_path / name
    assert target.is_dir()

    # Template files were copied.
    readme = target / "README.md"
    pyproject = target / "pyproject.toml"
    assert readme.is_file()
    assert pyproject.is_file()

    # Placeholders were resolved.
    readme_text = readme.read_text(encoding="utf-8")
    assert f"# {name}" in readme_text
    assert "<RECIPE_NAME>" not in readme_text
    assert "<OUTPUT_DIRECTORY>" not in readme_text
    assert f'name = "{name}"' in pyproject.read_text(encoding="utf-8")


def test_scaffold_refuses_existing_target(tmp_path):
    name = "already-here"
    (tmp_path / name).mkdir()

    ok = m.scaffold(name=name, output_dir=str(tmp_path))

    assert ok is False


# ---------------------------------------------------------------------------
# is_safe_recipe_name / path-traversal guard (Finding 2)
# ---------------------------------------------------------------------------


def test_is_safe_recipe_name():
    assert m.is_safe_recipe_name("my-recipe") is True
    assert m.is_safe_recipe_name("recipe_123") is True
    # Unsafe: separators, parent refs, absolute paths, empties.
    assert m.is_safe_recipe_name("../evil") is False
    assert m.is_safe_recipe_name("a/b") is False
    assert m.is_safe_recipe_name("a\\b") is False
    assert m.is_safe_recipe_name("..") is False
    assert m.is_safe_recipe_name(".") is False
    assert m.is_safe_recipe_name("") is False
    assert m.is_safe_recipe_name("/abs") is False


def test_scaffold_rejects_path_traversal(tmp_path):
    intended = tmp_path / "sub"
    intended.mkdir()

    ok = m.scaffold(name="../evil", output_dir=str(intended))

    assert ok is False
    # Crucially: nothing was written outside the intended output directory.
    assert not (tmp_path / "evil").exists()


# ---------------------------------------------------------------------------
# resilient placeholder replacement (Finding 1)
# ---------------------------------------------------------------------------


def test_scaffold_ignores_replace_placeholder_errors(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(m, "replace_in_file", boom)

    ok = m.scaffold(name="rec", output_dir=str(tmp_path))

    # A per-file replacement failure is warned about, not fatal ...
    assert ok is True
    # ... and the template files were still copied.
    assert (tmp_path / "rec" / "README.md").is_file()


# ---------------------------------------------------------------------------
# output_dir trailing-slash normalization (Finding 3)
# ---------------------------------------------------------------------------


def test_scaffold_normalizes_output_dir_trailing_slash(tmp_path):
    out = str(tmp_path) + "/"

    ok = m.scaffold(name="rec", output_dir=out)

    assert ok is True
    readme = (tmp_path / "rec" / "README.md").read_text(encoding="utf-8")
    # The trailing slash is stripped, so no doubled slash appears before the
    # recipe name in the rendered <OUTPUT_DIRECTORY>/<RECIPE_NAME>/ path.
    assert f"{tmp_path}/rec/" in readme
    assert "//rec/" not in readme


# ---------------------------------------------------------------------------
# missing templates directory (Finding 4)
# ---------------------------------------------------------------------------


def test_scaffold_fails_when_templates_missing(tmp_path, monkeypatch):
    real_isdir = os.path.isdir
    templates_suffix = os.path.join("resources", "templates")

    def fake_isdir(path):
        if str(path).endswith(templates_suffix):
            return False
        return real_isdir(path)

    monkeypatch.setattr(m.os.path, "isdir", fake_isdir)

    ok = m.scaffold(name="rec", output_dir=str(tmp_path))

    assert ok is False
    assert not (tmp_path / "rec").exists()


# ---------------------------------------------------------------------------
# junk / cache files must not leak into scaffolded recipes (report gap)
# ---------------------------------------------------------------------------


@pytest.fixture
def templates_with_junk():
    """Create cache junk inside the real templates dir, cleaned up after."""
    skill_dir = Path(m.__file__).resolve().parent.parent
    templates = skill_dir / "resources" / "templates"
    junk_files = [
        templates / ".ruff_cache" / "CACHEDIR.TAG",
        templates / "__pycache__" / "stale.pyc",
    ]
    for f in junk_files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"junk")
    try:
        yield templates
    finally:
        for f in junk_files:
            shutil.rmtree(f.parent, ignore_errors=True)


def test_scaffold_omits_cache_junk(tmp_path, templates_with_junk):
    ok = m.scaffold(name="rec", output_dir=str(tmp_path))

    assert ok is True
    target = tmp_path / "rec"
    leaked = [
        p
        for p in target.rglob("*")
        if p.name in {".ruff_cache", "__pycache__"} or p.suffix == ".pyc"
    ]
    assert leaked == []
