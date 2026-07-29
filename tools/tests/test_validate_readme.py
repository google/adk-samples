#!/usr/bin/env python3
"""Unit tests for tools/validate_readme.py."""

import textwrap
from pathlib import Path

import pytest
import validate_manifest as m
import validate_readme as r

# A minimal README that passes all checks (must be >= 100 words).
VALID_README = textwrap.dedent(
    """\
    # My Recipe

    This agent connects to an external API, processes incoming requests,
    and returns structured responses using Gemini. It demonstrates how to
    wire up a tool-calling loop with error handling and retry logic, making
    it a solid starting point for production integrations. The agent uses
    Vertex AI as its model backend and handles authentication automatically
    through Application Default Credentials. All configuration is driven by
    environment variables so the recipe works in local, CI, and deployed
    environments without code changes.

    ## Setup

    1. Create a Google Cloud project and enable the Vertex AI API.
    2. Authenticate with Application Default Credentials:

    ```bash
    gcloud auth application-default login
    ```

    3. Set the following environment variables:

    ```bash
    export GOOGLE_CLOUD_PROJECT=your-project-id
    export GOOGLE_CLOUD_LOCATION=us-central1
    ```

    ## Run

    ```bash
    uv run adk web
    ```
    """
)


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_recipe(
    root: Path,
    rel: str,
    readme: str | None = VALID_README,
) -> Path:
    """Create a recipe dir at root/rel with a README.

    is_recipe_dir() excludes README-only dirs, so we always write an
    agent.py alongside the README to ensure the dir qualifies as a recipe.
    When readme is None the README is omitted (tests for missing README).
    """
    recipe = root / rel
    recipe.mkdir(parents=True, exist_ok=True)
    # Always write agent.py so is_recipe_dir passes regardless of README state.
    _write(recipe / "agent.py", "# placeholder\n")
    if readme is not None:
        _write(recipe / "README.md", readme)
    return recipe


# ---------------------------------------------------------------------------
# validate_readme: valid input
# ---------------------------------------------------------------------------


def test_validate_readme_valid(tmp_path):
    readme = _write(tmp_path / "README.md", VALID_README)
    assert r.validate_readme(readme) == []


# ---------------------------------------------------------------------------
# validate_readme: description checks
# ---------------------------------------------------------------------------


def test_validate_readme_empty(tmp_path):
    readme = _write(tmp_path / "README.md", "")
    errors = r.validate_readme(readme)
    assert errors == ["README.md is empty"]


def test_validate_readme_todo_placeholder(tmp_path):
    content = VALID_README + "\nTODO: describe the agent here\n"
    readme = _write(tmp_path / "README.md", content)
    errors = r.validate_readme(readme)
    assert any("TODO" in e for e in errors)


def test_validate_readme_too_short(tmp_path):
    content = "# My Recipe\n\n## Setup\n\nRun it.\n\n## Run\n\n```bash\nuv run adk web\n```\n"
    readme = _write(tmp_path / "README.md", content)
    errors = r.validate_readme(readme)
    assert any("too short" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_readme: setup section checks
# ---------------------------------------------------------------------------


def test_validate_readme_missing_setup(tmp_path):
    # _SETUP_KEYWORDS matches on word boundaries, and is broad — "setup",
    # "prerequisites", "configuration", "installation", "requirements",
    # "environment" and more all count as a setup section. The replacement
    # heading has to dodge every one of them to reach the missing-setup
    # path at all.
    content = VALID_README.replace("## Setup", "## Overview of Dependencies")
    readme = _write(tmp_path / "README.md", content)
    errors = r.validate_readme(readme)
    assert any("setup" in e for e in errors)


def test_validate_readme_setup_synonym_prerequisites(tmp_path):
    content = VALID_README.replace("## Setup", "## Prerequisites")
    readme = _write(tmp_path / "README.md", content)
    assert r.validate_readme(readme) == []


def test_validate_readme_setup_synonym_installation(tmp_path):
    content = VALID_README.replace("## Setup", "## Installation")
    readme = _write(tmp_path / "README.md", content)
    assert r.validate_readme(readme) == []


def test_validate_readme_setup_synonym_getting_started(tmp_path):
    # "Getting Started" satisfies both setup and run keyword lists.
    content = VALID_README.replace("## Setup", "## Getting Started")
    readme = _write(tmp_path / "README.md", content)
    assert r.validate_readme(readme) == []


def test_validate_readme_setup_synonym_configuration(tmp_path):
    content = VALID_README.replace("## Setup", "## Configuration")
    readme = _write(tmp_path / "README.md", content)
    assert r.validate_readme(readme) == []


# ---------------------------------------------------------------------------
# validate_readme: run section checks
# ---------------------------------------------------------------------------


def test_validate_readme_missing_run(tmp_path):
    content = VALID_README.replace("## Run", "## Overview of Execution Details")
    readme = _write(tmp_path / "README.md", content)
    errors = r.validate_readme(readme)
    assert any("[run]" in e and "missing a run section" in e for e in errors)


def test_validate_readme_run_synonym_usage(tmp_path):
    content = VALID_README.replace("## Run", "## Usage")
    readme = _write(tmp_path / "README.md", content)
    assert r.validate_readme(readme) == []


def test_validate_readme_run_synonym_quickstart(tmp_path):
    content = VALID_README.replace("## Run", "## Quickstart")
    readme = _write(tmp_path / "README.md", content)
    assert r.validate_readme(readme) == []


def test_validate_readme_run_synonym_deploy(tmp_path):
    content = VALID_README.replace("## Run", "## Deploy")
    readme = _write(tmp_path / "README.md", content)
    assert r.validate_readme(readme) == []


def test_validate_readme_run_synonym_launch(tmp_path):
    content = VALID_README.replace("## Run", "## Launch")
    readme = _write(tmp_path / "README.md", content)
    assert r.validate_readme(readme) == []


def test_validate_readme_run_synonym_launching(tmp_path):
    content = VALID_README.replace("## Run", "## Launching the Agent")
    readme = _write(tmp_path / "README.md", content)
    assert r.validate_readme(readme) == []


# ---------------------------------------------------------------------------
# validate_readme: code block check
# ---------------------------------------------------------------------------


def test_validate_readme_no_code_block(tmp_path):
    import re as _re

    # Strip every fenced code block from the fixture.
    content = _re.sub(r"```.*?```", "", VALID_README, flags=_re.DOTALL)
    readme = _write(tmp_path / "README.md", content)
    errors = r.validate_readme(readme)
    assert any("code block" in e for e in errors)


# ---------------------------------------------------------------------------
# main (end-to-end against a fake tree)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    _make_recipe(tmp_path, "core/recipe-a")
    _make_recipe(tmp_path, "contrib/recipe-b")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(r, "REPO_ROOT", tmp_path)
    return tmp_path


def test_main_all_valid_returns_zero(fake_repo):
    assert r.main("core") == 0


def test_main_missing_readme_returns_one(tmp_path, monkeypatch):
    _make_recipe(tmp_path, "core/no-readme", readme=None)
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(r, "REPO_ROOT", tmp_path)
    assert r.main("core") == 1


def test_main_invalid_readme_returns_one(tmp_path, monkeypatch):
    _make_recipe(tmp_path, "core/bad", readme="too short\n")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(r, "REPO_ROOT", tmp_path)
    assert r.main("core/bad") == 1


def test_main_missing_readme_emits_github_annotation(
    tmp_path, monkeypatch, capsys
):
    _make_recipe(tmp_path, "core/no-readme", readme=None)
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(r, "REPO_ROOT", tmp_path)

    assert r.main("core") == 1

    out = capsys.readouterr().out
    assert "::error file=core/no-readme/README.md::" in out
    assert "missing" in out


def test_main_invalid_readme_emits_github_annotation(
    tmp_path, monkeypatch, capsys
):
    _make_recipe(tmp_path, "core/bad", readme="too short\n")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(r, "REPO_ROOT", tmp_path)

    assert r.main("core/bad") == 1

    out = capsys.readouterr().out
    assert "::error file=core/bad/README.md::" in out
