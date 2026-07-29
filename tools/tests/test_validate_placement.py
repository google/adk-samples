#!/usr/bin/env python3
"""Unit tests for tools/validate_placement.py.

`describe_violation` is pure string logic and gets a table. The scanning
layer is exercised against temporary trees, since the whole point of the
checker is what it finds on disk.
"""

from pathlib import Path

import pytest
import validate_placement as m

MANIFEST = "manifest.yaml"


def _make_manifest(root: Path, rel_dir: str) -> Path:
    """Create <root>/<rel_dir>/manifest.yaml and return the manifest path."""
    directory = root / rel_dir
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / MANIFEST
    manifest.write_text("type: recipe\n", encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# describe_violation — pure string logic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_parts",
    [
        ["skills", "retail", "store-ops", MANIFEST],
        ["skills", "hr", "onboarding", MANIFEST],
        ["skills", "finance", "month-end-close", MANIFEST],
    ],
)
def test_correctly_placed_returns_none(rel_parts):
    assert m.describe_violation(rel_parts) is None


def test_flat_skill_is_rejected():
    """The case that matters: a solution dropped straight under skills/."""
    message = m.describe_violation(["skills", "foo", MANIFEST])
    assert message is not None
    assert "no vertical" in message
    assert "skills/<vertical>/foo" in message


def test_flat_skill_message_names_the_offending_dir():
    message = m.describe_violation(["skills", "store-ops", MANIFEST])
    assert "'skills/store-ops'" in message
    assert "skills/retail/store-ops" in message


def test_manifest_at_root_of_skills_is_rejected():
    message = m.describe_violation(["skills", MANIFEST])
    assert message is not None
    assert "no vertical" in message


def test_too_deep_is_rejected():
    message = m.describe_violation(
        ["skills", "retail", "store-ops", "subproject", MANIFEST]
    )
    assert message is not None
    assert "too deeply" in message
    assert "skills/retail/store-ops/subproject" in message


# ---------------------------------------------------------------------------
# find_manifests — scanning and pruning
# ---------------------------------------------------------------------------


def test_find_manifests_on_missing_root_is_empty(tmp_path):
    assert m.find_manifests(tmp_path / "skills") == []


def test_find_manifests_prunes_vendored_dirs(tmp_path):
    _make_manifest(tmp_path, "skills/retail/store-ops")
    _make_manifest(tmp_path, "skills/retail/store-ops/.venv/lib/pkg")
    _make_manifest(tmp_path, "skills/retail/store-ops/node_modules/thing")
    found = m.find_manifests(tmp_path / "skills")
    assert [str(p.relative_to(tmp_path)) for p in found] == [
        f"skills/retail/store-ops/{MANIFEST}"
    ]


# ---------------------------------------------------------------------------
# check_root — scanning plus classification
# ---------------------------------------------------------------------------


def test_check_root_accepts_valid_tree(tmp_path, capsys):
    _make_manifest(tmp_path, "skills/retail/store-ops")
    _make_manifest(tmp_path, "skills/hr/onboarding")
    assert m.check_root("skills", repo_root=tmp_path) == []
    assert "::error" not in capsys.readouterr().out


def test_check_root_flags_flat_skill(tmp_path, capsys):
    _make_manifest(tmp_path, "skills/retail/store-ops")
    _make_manifest(tmp_path, "skills/oops")
    errors = m.check_root("skills", repo_root=tmp_path)
    assert len(errors) == 1
    out = capsys.readouterr().out
    assert f"::error file=skills/oops/{MANIFEST}::" in out


def test_check_root_flags_every_offender(tmp_path):
    _make_manifest(tmp_path, "skills/a")
    _make_manifest(tmp_path, "skills/b")
    _make_manifest(tmp_path, "skills/retail/ok")
    assert len(m.check_root("skills", repo_root=tmp_path)) == 2


def test_check_root_flags_too_deep(tmp_path):
    _make_manifest(tmp_path, "skills/retail/store-ops/inner")
    errors = m.check_root("skills", repo_root=tmp_path)
    assert len(errors) == 1
    assert "too deeply" in errors[0]


def test_check_root_on_empty_root(tmp_path):
    (tmp_path / "skills").mkdir()
    assert m.check_root("skills", repo_root=tmp_path) == []


# ---------------------------------------------------------------------------
# main — exit codes against the real repository
# ---------------------------------------------------------------------------


def test_main_passes_on_the_real_repo():
    """The repository must always be placement-clean on main."""
    assert m.main() == 0


def test_main_tolerates_a_missing_skills_root(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    assert m.main() == 0
    assert "[SKIP]" in capsys.readouterr().out


def test_main_reports_failure(monkeypatch, tmp_path, capsys):
    _make_manifest(tmp_path, "skills/oops")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    assert m.main() == 1
    out = capsys.readouterr().out
    assert "ACTION REQUIRED" in out
    assert "1 misplaced recipe" in out


def test_main_narrows_to_the_given_scope(monkeypatch, tmp_path):
    """A scope must not surface violations from elsewhere in the tree, or
    `uv run validate <a-core-recipe>` would fail on an unrelated skill."""
    _make_manifest(tmp_path, "skills/oops")
    _make_manifest(tmp_path, "skills/retail/store-ops")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    assert m.main("skills/retail") == 0
    assert m.main("skills") == 1
    assert m.main() == 1


def test_main_ignores_a_scope_outside_the_checked_roots(
    monkeypatch, tmp_path, capsys
):
    _make_manifest(tmp_path, "skills/oops")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    assert m.main("core/python/foo") == 0
    assert "::error" not in capsys.readouterr().out


def test_registered_as_a_validate_subcommand():
    import validate

    assert "placement" in validate.SUBCOMMANDS
    assert validate.SUBCOMMANDS["placement"][1] is m.main
    assert "placement" in validate.VALID_SUBCOMMANDS


# ---------------------------------------------------------------------------
# Configuration guards
# ---------------------------------------------------------------------------


def test_checked_roots_tracks_the_shared_constant():
    import validate_manifest as vm

    assert set(m.CHECKED_ROOTS) == set(vm.NAMESPACE_REQUIRED_ROOTS)
    assert "skills" in m.CHECKED_ROOTS


def test_core_and_contrib_are_not_checked_yet():
    """Flat core/<recipe> is still legal, so those roots stay out of scope
    until that layout is retired."""
    assert "core" not in m.CHECKED_ROOTS
    assert "contrib" not in m.CHECKED_ROOTS
