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
"""Unit tests for check_lockfile_hashes.py.

A malformed uv.lock used to raise AttributeError here, after which the
workflow announced a confident and wrong diagnosis about missing sha256
hashes. These tests pin both halves of the fix: the shape is reported as
what it is, and a missing hash now carries a rule and a fix instead of a
bare `name==version: url` line with no annotation at all.
"""

import sys
from pathlib import Path

import check_lockfile_hashes as m

# The numbers the workflow branches on. Spelled out rather than imported so
# a change to them has to be made deliberately in two places.
EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_CI_FAULT = 2

_HASH = "sha256:" + "0" * 64


def _lock(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "uv.lock"
    path.write_text(body, encoding="utf-8")
    return path


def _run(path, monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", ["check_lockfile_hashes.py", str(path)])
    return m.main()


def test_hashed_lockfile_passes(tmp_path, monkeypatch, capsys):
    path = _lock(
        tmp_path,
        "version = 1\n\n"
        "[[package]]\n"
        'name = "foo"\nversion = "1.0"\n\n'
        "[[package.wheels]]\n"
        'url = "https://files.pythonhosted.org/foo-1.0.whl"\n'
        f'hash = "{_HASH}"\n',
    )
    assert _run(path, monkeypatch) == EXIT_OK
    assert "::error" not in capsys.readouterr().out


def test_missing_hash_is_annotated_with_a_rule_and_a_fix(
    tmp_path, monkeypatch, capsys
):
    """It used to print `  foo==1.0: <url>` — no verb, no rule, no fix, and
    no annotation, so the PR Files tab showed nothing at all."""
    path = _lock(
        tmp_path,
        "version = 1\n\n"
        "[[package]]\n"
        'name = "foo"\nversion = "1.0"\n\n'
        "[[package.wheels]]\n"
        'url = "https://files.pythonhosted.org/foo-1.0.whl"\n',
    )
    assert _run(path, monkeypatch) == EXIT_VIOLATIONS
    out = capsys.readouterr().out
    assert "foo==1.0" in out
    assert "uv lock" in out
    assert f"::error file={path}::" in out


def test_malformed_hash_is_reported(tmp_path, monkeypatch, capsys):
    path = _lock(
        tmp_path,
        "version = 1\n\n"
        "[[package]]\n"
        'name = "foo"\nversion = "1.0"\n\n'
        "[[package.wheels]]\n"
        'url = "https://files.pythonhosted.org/foo-1.0.whl"\n'
        'hash = "md5:deadbeef"\n',
    )
    assert _run(path, monkeypatch) == EXIT_VIOLATIONS
    assert "unrecognised hash" in capsys.readouterr().out


def test_sourceless_kinds_are_skipped(tmp_path, monkeypatch):
    path = _lock(
        tmp_path,
        "version = 1\n\n"
        "[[package]]\n"
        'name = "my-recipe"\nversion = "0.1.0"\n'
        'source = { virtual = "." }\n',
    )
    assert _run(path, monkeypatch) == EXIT_OK


def test_a_package_entry_that_is_not_a_table_is_reported_not_crashed(
    tmp_path, monkeypatch, capsys
):
    path = _lock(tmp_path, 'version = 1\npackage = ["not-a-table"]\n')
    assert _run(path, monkeypatch) == EXIT_VIOLATIONS
    out = capsys.readouterr().out
    assert "AttributeError" not in out
    assert "entry #1" in out
    assert "not a table" in out
    assert f"::error file={path}::" in out


def test_a_non_table_distribution_is_reported_not_crashed(
    tmp_path, monkeypatch, capsys
):
    path = _lock(
        tmp_path,
        "version = 1\n\n"
        "[[package]]\n"
        'name = "foo"\nversion = "1.0"\nwheels = ["a-wheel"]\n',
    )
    assert _run(path, monkeypatch) == EXIT_VIOLATIONS
    out = capsys.readouterr().out
    assert "AttributeError" not in out
    assert "not a table" in out


def test_package_declared_as_a_table_is_reported_not_crashed(
    tmp_path, monkeypatch, capsys
):
    path = _lock(tmp_path, 'version = 1\n\n[package]\nname = "foo"\n')
    assert _run(path, monkeypatch) == EXIT_VIOLATIONS
    assert "not an array of tables" in capsys.readouterr().out


def test_invalid_toml_is_reported_against_the_lockfile(
    tmp_path, monkeypatch, capsys
):
    path = _lock(tmp_path, "[[package\nname = broken\n")
    assert _run(path, monkeypatch) == EXIT_VIOLATIONS
    out = capsys.readouterr().out
    assert "not valid TOML" in out
    assert "Traceback" not in out


def test_a_missing_lockfile_is_a_ci_fault(tmp_path, monkeypatch, capsys):
    """The path comes from the workflow's discovery step, not the recipe."""
    assert _run(tmp_path / "absent.lock", monkeypatch) == EXIT_CI_FAULT
    out = capsys.readouterr().out
    assert "[ci-fault]" in out
    assert "::error file=" not in out


def test_wrong_argument_count_is_a_ci_fault(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_lockfile_hashes.py"])
    assert m.main() == EXIT_CI_FAULT
    assert "[ci-fault]" in capsys.readouterr().out


def test_an_unexpected_crash_is_a_ci_fault(tmp_path, monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise RuntimeError("checker bug")

    monkeypatch.setattr(m, "_check_package", boom)
    path = _lock(
        tmp_path, 'version = 1\n\n[[package]]\nname = "foo"\nversion = "1"\n'
    )
    assert _run(path, monkeypatch) == EXIT_CI_FAULT
    out = capsys.readouterr().out
    assert "RuntimeError: checker bug" in out
    assert "::error file=" not in out
