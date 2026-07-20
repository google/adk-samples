#!/usr/bin/env python3
"""Unit tests for tools/validate_structure.py.

Structure similar to test_validate_manifest.py: tiny per-function tests
plus a handful of end-to-end tests that monkeypatch REPO_ROOT to a fake
recipe tree so we exercise the real orchestration without touching the
committed recipes.
"""

import textwrap
from pathlib import Path

import pytest
import validate_manifest as vm
import validate_structure as m

# Shared minimal-valid manifest — mirrors test_validate_manifest.VALID_MANIFEST.
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


def _make_python_recipe(
    root: Path,
    rel: str,
    *,
    manifest: str | None = VALID_MANIFEST,
    include_agents: bool = False,
    include_python_files: bool = True,
) -> Path:
    """Create a directory that looks like a real Python recipe: manifest
    + README + pyproject.toml + uv.lock + .env.example + tests/. Callers
    flip flags to remove pieces and exercise negative paths."""
    recipe = root / rel
    recipe.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        _write(recipe / "manifest.yaml", manifest)
    _write(recipe / "README.md", "# recipe\n")
    if include_python_files:
        _write(recipe / "pyproject.toml", "[project]\nname='x'\n")
        _write(recipe / "uv.lock", "# lockfile\n")
        _write(recipe / ".env.example", "FOO=1\n")
        _write(recipe / "tests" / "test_runnability.py", "def test(): pass\n")
    if include_agents:
        _write(recipe / "AGENTS.md", "# agents\n")
    return recipe


# ---------------------------------------------------------------------------
# required_files_for
# ---------------------------------------------------------------------------


def _base_policy() -> dict:
    # Mirrors the intended shape of .github/policy.yml — manifest.yaml is
    # NOT in `always`: check 1 in validate_recipe is authoritative for it,
    # and including it here would produce a duplicate error report.
    return {
        "required_files": {
            "always": ["README.md"],
            "by_root": {
                "core": ["AGENTS.md"],
                "contrib": [],
                "skills": ["SKILL.md"],
            },
            "by_language": {
                "python": ["pyproject.toml", "uv.lock"],
                "java": [],
            },
        }
    }


def test_required_files_for_core_python():
    files = m.required_files_for(_base_policy(), "core", "python")
    assert files == [
        "README.md",
        "AGENTS.md",
        "pyproject.toml",
        "uv.lock",
    ]


def test_required_files_for_contrib_python():
    files = m.required_files_for(_base_policy(), "contrib", "python")
    # No AGENTS.md — contrib has no by_root files.
    assert files == ["README.md", "pyproject.toml", "uv.lock"]


def test_required_files_for_skills_no_language():
    files = m.required_files_for(_base_policy(), "skills", None)
    assert files == ["README.md", "SKILL.md"]


def test_required_files_for_unknown_root_and_language():
    files = m.required_files_for(_base_policy(), "core", "brainfuck")
    # Unknown language falls back to only always + by_root[core].
    assert files == ["README.md", "AGENTS.md"]


def test_required_files_for_missing_config():
    # An empty policy shouldn't crash — it should return an empty list.
    assert m.required_files_for({}, "core", "python") == []


def test_required_files_for_deduplicates():
    # The dedup path is exercised even if the intended policy layout
    # avoids overlaps — someone editing policy.yml could reintroduce one.
    policy = {
        "required_files": {
            "always": ["README.md"],
            "by_root": {"core": ["README.md", "AGENTS.md"]},
            "by_language": {"python": ["AGENTS.md", "pyproject.toml"]},
        }
    }
    files = m.required_files_for(policy, "core", "python")
    assert files == ["README.md", "AGENTS.md", "pyproject.toml"]


# ---------------------------------------------------------------------------
# detect_language / is_large_recipe / recipe_root_of
# ---------------------------------------------------------------------------


def test_detect_language_valid(tmp_path):
    manifest = _write(tmp_path / "manifest.yaml", VALID_MANIFEST)
    assert m.detect_language(manifest) == "python"


def test_detect_language_missing_manifest(tmp_path):
    assert m.detect_language(tmp_path / "missing.yaml") is None


def test_detect_language_bad_yaml(tmp_path):
    manifest = _write(tmp_path / "manifest.yaml", "language: [unclosed\n")
    assert m.detect_language(manifest) is None


def test_detect_language_no_language_field(tmp_path):
    manifest = _write(tmp_path / "manifest.yaml", "type: standalone\n")
    assert m.detect_language(manifest) is None


def test_detect_language_lowercases(tmp_path):
    # Schema disallows uppercase, but detect_language should still
    # normalise — the schema check reports the violation separately.
    manifest = _write(tmp_path / "manifest.yaml", "language: PYTHON\n")
    assert m.detect_language(manifest) == "python"


def test_is_large_recipe_true(tmp_path):
    manifest = _write(
        tmp_path / "manifest.yaml",
        VALID_MANIFEST + "large: true\n",
    )
    assert m.is_large_recipe(manifest) is True


def test_is_large_recipe_false_by_default(tmp_path):
    manifest = _write(tmp_path / "manifest.yaml", VALID_MANIFEST)
    assert m.is_large_recipe(manifest) is False


def test_is_large_recipe_missing_manifest(tmp_path):
    assert m.is_large_recipe(tmp_path / "missing.yaml") is False


def test_recipe_root_of_core(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    (tmp_path / "core" / "recipe-a").mkdir(parents=True)
    assert m.recipe_root_of(tmp_path / "core" / "recipe-a") == "core"


def test_recipe_root_of_namespaced(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    (tmp_path / "contrib" / "python" / "recipe-b").mkdir(parents=True)
    assert (
        m.recipe_root_of(tmp_path / "contrib" / "python" / "recipe-b")
        == "contrib"
    )


# ---------------------------------------------------------------------------
# check_folder_name
# ---------------------------------------------------------------------------


def test_check_folder_name_valid(tmp_path):
    d = tmp_path / "my-recipe"
    d.mkdir()
    assert m.check_folder_name(d, max_length=30) == []


def test_check_folder_name_rejects_underscore(tmp_path):
    d = tmp_path / "my_recipe"
    d.mkdir()
    errs = m.check_folder_name(d, max_length=30)
    assert errs and "invalid" in errs[0].lower()


def test_check_folder_name_rejects_uppercase(tmp_path):
    d = tmp_path / "MyRecipe"
    d.mkdir()
    errs = m.check_folder_name(d, max_length=30)
    assert errs and "invalid" in errs[0].lower()


def test_check_folder_name_rejects_leading_digit(tmp_path):
    d = tmp_path / "1recipe"
    d.mkdir()
    errs = m.check_folder_name(d, max_length=30)
    assert errs and "invalid" in errs[0].lower()


def test_check_folder_name_too_long(tmp_path):
    d = tmp_path / ("x" * 31)
    d.mkdir()
    errs = m.check_folder_name(d, max_length=30)
    assert errs and "31 characters" in errs[0]


def test_check_folder_name_both_violations_reported(tmp_path):
    # Invalid AND too long → both errors surface at once so the maintainer
    # doesn't have to iterate twice.
    d = tmp_path / ("X" * 31)
    d.mkdir()
    errs = m.check_folder_name(d, max_length=30)
    assert len(errs) == 2


# ---------------------------------------------------------------------------
# check_required_files
# ---------------------------------------------------------------------------


def test_check_required_files_all_present(tmp_path):
    recipe = _make_python_recipe(tmp_path, "core/foo", include_agents=True)
    policy = _base_policy()
    assert m.check_required_files(recipe, "core", "python", policy) == []


def test_check_required_files_missing_agents_in_core(tmp_path):
    recipe = _make_python_recipe(tmp_path, "core/foo", include_agents=False)
    policy = _base_policy()
    errs = m.check_required_files(recipe, "core", "python", policy)
    assert any("AGENTS.md" in e for e in errs)


def test_check_required_files_contrib_does_not_require_agents(tmp_path):
    recipe = _make_python_recipe(tmp_path, "contrib/foo", include_agents=False)
    policy = _base_policy()
    assert m.check_required_files(recipe, "contrib", "python", policy) == []


def test_check_required_files_nested_path(tmp_path):
    # tests/test_runnability.py is a nested requirement — the checker
    # must look at the actual file, not just the directory.
    recipe = _make_python_recipe(tmp_path, "core/foo", include_agents=True)
    # Remove the file but keep tests/ as a directory.
    (recipe / "tests" / "test_runnability.py").unlink()
    policy = _base_policy()
    policy["required_files"]["by_language"]["python"].append(
        "tests/test_runnability.py"
    )
    errs = m.check_required_files(recipe, "core", "python", policy)
    assert any("tests/test_runnability.py" in e for e in errs)


def test_check_required_files_directory_does_not_satisfy(tmp_path):
    # If a required file exists as a DIRECTORY, that must still count as
    # missing (is_file() guards this). Use AGENTS.md so the check runs
    # even with manifest.yaml removed from `always` in the real policy.
    recipe = tmp_path / "core" / "foo"
    (recipe / "AGENTS.md").mkdir(parents=True)
    _write(recipe / "README.md", "# x\n")
    policy = _base_policy()
    errs = m.check_required_files(recipe, "core", None, policy)
    assert any("AGENTS.md" in e for e in errs)


# ---------------------------------------------------------------------------
# check_size_and_count
# ---------------------------------------------------------------------------


def _size_policy(max_files: int = 10, max_size_mb: int = 1) -> dict:
    """Minimal policy with tight limits so tests can trip them with
    just a few files."""
    return {
        "recipe_size_limits": {
            "core": {
                "default": {"max_files": max_files, "max_size_mb": max_size_mb},
                "large": {"max_files": 1000, "max_size_mb": 100},
            },
            "contrib": {
                "default": {"max_files": max_files, "max_size_mb": max_size_mb},
            },
        },
        "excluded_paths": {
            "python": {"dirs": ["__pycache__"], "files": ["uv.lock"]},
        },
    }


def test_check_size_and_count_under_limits(tmp_path):
    recipe = _make_python_recipe(tmp_path, "core/foo", include_agents=True)
    manifest = recipe / "manifest.yaml"
    assert (
        m.check_size_and_count(recipe, "core", _size_policy(), manifest) == []
    )


def test_check_size_and_count_exceeds_file_count(tmp_path):
    recipe = _make_python_recipe(tmp_path, "core/foo", include_agents=True)
    manifest = recipe / "manifest.yaml"
    # Add enough files to exceed max_files=10 (recipe already has ~6).
    for i in range(20):
        _write(recipe / f"extra_{i}.py", "# noop\n")
    errs = m.check_size_and_count(recipe, "core", _size_policy(), manifest)
    assert any("exceeding the limit" in e for e in errs)


def test_check_size_and_count_exceeds_size(tmp_path):
    recipe = _make_python_recipe(tmp_path, "core/foo", include_agents=True)
    manifest = recipe / "manifest.yaml"
    # Write a single 2 MiB blob; limit is 1 MiB.
    _write(recipe / "blob.bin", "x" * (2 * 1024 * 1024))
    errs = m.check_size_and_count(recipe, "core", _size_policy(), manifest)
    assert any("exceeds" in e and "MB" in e for e in errs)


def test_check_size_and_count_excludes_uv_lock(tmp_path):
    recipe = _make_python_recipe(tmp_path, "core/foo", include_agents=True)
    manifest = recipe / "manifest.yaml"
    # Write a 2 MiB uv.lock (excluded) — should NOT trip the size limit.
    _write(recipe / "uv.lock", "y" * (2 * 1024 * 1024))
    assert (
        m.check_size_and_count(recipe, "core", _size_policy(), manifest) == []
    )


def test_check_size_and_count_excludes_pycache_dir(tmp_path):
    recipe = _make_python_recipe(tmp_path, "core/foo", include_agents=True)
    manifest = recipe / "manifest.yaml"
    # A pruned dir with many files must not count toward the file limit.
    for i in range(50):
        _write(recipe / "app" / "__pycache__" / f"m_{i}.pyc", "x")
    assert (
        m.check_size_and_count(recipe, "core", _size_policy(), manifest) == []
    )


def test_check_size_and_count_large_tier_relaxes(tmp_path):
    # A recipe that would fail under `default` should pass under `large`
    # when the manifest opts in.
    recipe = _make_python_recipe(
        tmp_path,
        "core/foo",
        manifest=VALID_MANIFEST + "large: true\n",
        include_agents=True,
    )
    manifest = recipe / "manifest.yaml"
    for i in range(50):
        _write(recipe / f"extra_{i}.py", "x")
    assert (
        m.check_size_and_count(recipe, "core", _size_policy(), manifest) == []
    )


def test_check_size_and_count_root_without_limits_skips(tmp_path):
    # No `skills` section in recipe_size_limits → nothing to enforce.
    recipe = _make_python_recipe(tmp_path, "skills/foo", include_agents=False)
    manifest = recipe / "manifest.yaml"
    assert (
        m.check_size_and_count(recipe, "skills", _size_policy(), manifest) == []
    )


def test_check_size_and_count_missing_large_tier_fails(tmp_path):
    # If a recipe opts into `large: true` but the root has no `large` tier
    # in policy.yml, the check must FAIL with a clear message (previously
    # it silently fell back to `default`, leaving the author unaware).
    recipe = _make_python_recipe(
        tmp_path,
        "core/foo",
        manifest=VALID_MANIFEST + "large: true\n",
        include_agents=True,
    )
    manifest = recipe / "manifest.yaml"
    # Policy with `default` but no `large` tier for core.
    policy = {
        "recipe_size_limits": {
            "core": {"default": {"max_files": 50, "max_size_mb": 5}}
        }
    }
    errs = m.check_size_and_count(recipe, "core", policy, manifest)
    assert errs
    assert any(
        "manifest.large=true" in e and "does not define a 'large' tier" in e
        for e in errs
    )


def test_check_size_and_count_missing_both_tiers_is_a_noop(tmp_path):
    # Edge case: root_limits exists but has neither `default` nor `large`.
    # A `large: true` request without a matching tier still fails; a
    # default (no manifest.large) request finds nothing to enforce and
    # returns [] — same shape as "no root_limits at all".
    recipe = _make_python_recipe(tmp_path, "core/foo", include_agents=True)
    manifest = recipe / "manifest.yaml"
    policy = {"recipe_size_limits": {"core": {}}}
    assert m.check_size_and_count(recipe, "core", policy, manifest) == []


# ---------------------------------------------------------------------------
# validate_recipe (end-to-end orchestration on a fake tree)
# ---------------------------------------------------------------------------


def _full_policy() -> dict:
    """Realistic policy combining required_files + size limits +
    excluded paths — mirrors the shape of the committed policy.yml."""
    p = _base_policy()
    p.update(_size_policy(max_files=200, max_size_mb=50))
    p["recipe_naming"] = {"max_folder_name_length": 30}
    p["required_files"]["by_language"]["python"] = [
        "pyproject.toml",
        "uv.lock",
        ".env.example",
        "tests/test_runnability.py",
    ]
    return p


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    """Point validate_structure's REPO_ROOT at a scratch dir so the
    recipe_root_of() calls used by validate_recipe/main resolve
    correctly. Independent of the `fake_repo` fixture below, which
    additionally writes a policy.yml for main() to load."""
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    return tmp_path


def test_validate_recipe_full_pass(isolated_repo):
    recipe = _make_python_recipe(
        isolated_repo, "core/good", include_agents=True
    )
    schema = vm.load_schema()
    assert m.validate_recipe(recipe, _full_policy(), schema) == []


def test_validate_recipe_missing_manifest_reports_and_skips_schema(
    isolated_repo,
):
    recipe = _make_python_recipe(
        isolated_repo, "core/bad", manifest=None, include_agents=True
    )
    schema = vm.load_schema()
    errs = m.validate_recipe(recipe, _full_policy(), schema)
    # Missing manifest is reported…
    assert any("manifest" in e and "missing" in e for e in errs)
    # …and required files that depend on the manifest's language are
    # NOT reported (we don't know the language, so we skip that source).
    # But `always` + `by_root[core]` files still apply.
    # AGENTS.md is present, README is present, so no other errors expected
    # beyond the manifest itself.
    assert not any(".env.example" in e for e in errs)


def test_validate_recipe_invalid_manifest_still_runs_other_checks(
    isolated_repo,
):
    # Manifest exists but is schema-invalid (bad type). detect_language
    # still returns "python", so by_language files are still enforced.
    bad_manifest = textwrap.dedent(
        """\
        type: not_a_valid_type
        status: active
        language: python
        description: A valid description here.
        ownership:
          team: My Team
          poc: my-github-id
        """
    )
    recipe = _make_python_recipe(
        isolated_repo, "core/foo", manifest=bad_manifest, include_agents=True
    )
    # Remove .env.example so we can assert the language-based check ran.
    (recipe / ".env.example").unlink()
    schema = vm.load_schema()
    errs = m.validate_recipe(recipe, _full_policy(), schema)
    assert any("manifest" in e for e in errs)
    assert any(".env.example" in e for e in errs)


def test_validate_recipe_bad_folder_name(isolated_repo):
    recipe = _make_python_recipe(
        isolated_repo, "core/Bad_Name", include_agents=True
    )
    schema = vm.load_schema()
    errs = m.validate_recipe(recipe, _full_policy(), schema)
    assert any("folder-name" in e for e in errs)


def test_validate_recipe_missing_agents_in_core(isolated_repo):
    recipe = _make_python_recipe(
        isolated_repo, "core/foo", include_agents=False
    )
    schema = vm.load_schema()
    errs = m.validate_recipe(recipe, _full_policy(), schema)
    assert any("AGENTS.md" in e for e in errs)


def test_validate_recipe_outside_known_roots(isolated_repo):
    # A recipe under some other top-level dir the checker doesn't recognise.
    recipe = _make_python_recipe(
        isolated_repo, "playground/foo", include_agents=True
    )
    schema = vm.load_schema()
    errs = m.validate_recipe(recipe, _full_policy(), schema)
    assert errs and "must live under" in errs[0]


# ---------------------------------------------------------------------------
# main (end-to-end against a fake tree)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A fake repo with a mix of good and bad recipes so main() has
    something to iterate over."""
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(m, "POLICY_PATH", tmp_path / ".github" / "policy.yml")
    monkeypatch.setattr(vm, "REPO_ROOT", tmp_path)
    # Write a policy.yml matching _full_policy() for main() to load.
    import yaml as _yaml

    _write(tmp_path / ".github" / "policy.yml", _yaml.safe_dump(_full_policy()))
    return tmp_path


def test_main_all_recipes_pass(fake_repo):
    _make_python_recipe(fake_repo, "core/good-a", include_agents=True)
    _make_python_recipe(fake_repo, "core/good-b", include_agents=True)
    assert m.main("core") == 0


def test_main_returns_one_on_failure(fake_repo, capsys):
    _make_python_recipe(fake_repo, "core/good", include_agents=True)
    _make_python_recipe(fake_repo, "core/broken", include_agents=False)
    assert m.main("core") == 1
    out = capsys.readouterr().out
    assert "::error file=core/broken::" in out
    assert "AGENTS.md" in out


def test_main_single_recipe_scope(fake_repo):
    _make_python_recipe(fake_repo, "core/only", include_agents=True)
    assert m.main("core/only") == 0
