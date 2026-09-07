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
"""Unit tests for the generate-python-runnability-test skill script.

Focused on import-support detection: the generated test does
``import <module>``, which only resolves when the RECIPE ROOT is on
``sys.path``. That was previously assumed rather than checked, so a recipe
with no ``[build-system]`` got a test that always died with
``ModuleNotFoundError`` while ``py_compile`` still reported success.
"""

import ast
from pathlib import Path

import generate_runnability_test as m


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _recipe(tmp_path: Path, pyproject: str, pkg: str = "app") -> Path:
    _write(tmp_path / pkg / "agent.py", "root_agent = object()\n")
    _write(tmp_path / "pyproject.toml", pyproject)
    return tmp_path


# ---------------------------------------------------------------------------
# detect_import_support
# ---------------------------------------------------------------------------


def test_build_system_means_installable(tmp_path):
    recipe = _recipe(
        tmp_path,
        '[project]\nname = "x"\n'
        '[build-system]\nrequires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n',
    )
    assert m.detect_import_support(recipe) == "installable"


def test_pythonpath_ini_covering_root(tmp_path):
    recipe = _recipe(
        tmp_path,
        '[project]\nname = "x"\n'
        '[tool.pytest.ini_options]\npythonpath = ["."]\n',
    )
    assert m.detect_import_support(recipe) == "pythonpath-ini"


def test_pythonpath_ini_not_covering_root(tmp_path):
    recipe = _recipe(
        tmp_path,
        '[project]\nname = "x"\n'
        '[tool.pytest.ini_options]\npythonpath = ["src"]\n',
    )
    assert m.detect_import_support(recipe) == "unresolved"


def test_root_conftest_is_sufficient_even_when_empty(tmp_path):
    # Verified against real pytest: under prepend import mode a conftest's own
    # directory goes on sys.path, and for a ROOT conftest that is the recipe
    # root — so its contents are irrelevant.
    recipe = _recipe(tmp_path, '[project]\nname = "x"\n')
    _write(recipe / "conftest.py", "")
    assert m.detect_import_support(recipe) == "existing-conftest"


def test_tests_conftest_counts_only_when_it_touches_syspath(tmp_path):
    recipe = _recipe(tmp_path, '[project]\nname = "x"\n')
    _write(recipe / "tests" / "conftest.py", "# just fixtures\n")
    # tests/conftest.py only adds <recipe>/tests, which does not help.
    assert m.detect_import_support(recipe) == "unresolved"

    # A mere MENTION of sys.path in prose must not count as import support.
    _write(
        recipe / "tests" / "conftest.py",
        '"""Fixtures. Deliberately does not touch sys.path."""\n',
    )
    assert m.detect_import_support(recipe) == "unresolved"

    _write(
        recipe / "tests" / "conftest.py",
        "import sys\nsys.path.insert(0, '..')\n",
    )
    assert m.detect_import_support(recipe) == "existing-conftest"


def test_missing_pyproject_is_unresolved(tmp_path):
    _write(tmp_path / "app" / "agent.py", "root_agent = object()\n")
    assert m.detect_import_support(tmp_path) == "unresolved"


def test_unparseable_pyproject_falls_back_to_text_scan(tmp_path):
    # Malformed TOML must not be read as "no build system" — the text scan
    # is the safety net.
    recipe = _recipe(
        tmp_path,
        '[project\nname = "x"\n[build-system]\nrequires = ["hatchling"]\n',
    )
    assert m.detect_import_support(recipe) == "installable"


# ---------------------------------------------------------------------------
# conftest generation
# ---------------------------------------------------------------------------


def test_generates_conftest_when_import_would_fail(tmp_path):
    recipe = _recipe(tmp_path, '[project]\nname = "x"\n', pkg="scripts")

    report = m.run(recipe, None, dry_run=False, overwrite=False)

    assert report.action == "wrote"
    assert report.import_support == "generated-conftest"
    assert report.conftest_action == "wrote"
    conftest = recipe / "tests" / "conftest.py"
    assert conftest.is_file()
    content = conftest.read_text(encoding="utf-8")
    assert "sys.path.insert" in content
    ast.parse(content)  # must be valid Python
    assert any("build-system" in w for w in report.warnings)


def test_no_conftest_generated_when_installable(tmp_path):
    recipe = _recipe(
        tmp_path,
        '[project]\nname = "x"\n'
        '[build-system]\nrequires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n',
    )

    report = m.run(recipe, None, dry_run=False, overwrite=False)

    assert report.import_support == "installable"
    assert report.conftest_action is None
    assert not (recipe / "tests" / "conftest.py").exists()
    assert report.warnings == []


def test_existing_conftest_is_never_clobbered(tmp_path):
    recipe = _recipe(tmp_path, '[project]\nname = "x"\n')
    original = "# hand-written, does not touch sys.path\n"
    _write(recipe / "tests" / "conftest.py", original)

    report = m.run(recipe, None, dry_run=False, overwrite=False)

    assert report.conftest_action == "skipped"
    assert (recipe / "tests" / "conftest.py").read_text() == original
    assert any("left alone" in w for w in report.warnings)


def test_dry_run_writes_nothing(tmp_path):
    recipe = _recipe(tmp_path, '[project]\nname = "x"\n')

    report = m.run(recipe, None, dry_run=True, overwrite=False)

    assert report.action == "would_write"
    assert report.conftest_action == "would_write"
    assert not (recipe / "tests" / "conftest.py").exists()
    assert not (recipe / "tests" / "test_runnability.py").exists()


def test_report_json_includes_new_fields(tmp_path):
    import json

    recipe = _recipe(tmp_path, '[project]\nname = "x"\n')
    report = m.run(recipe, None, dry_run=True, overwrite=False)

    payload = json.loads(report.to_json())
    assert payload["import_support"] == "generated-conftest"
    assert payload["conftest_action"] == "would_write"
    assert isinstance(payload["warnings"], list)
