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
"""Unit tests for the extract-python-environment-variables skill script."""

import ast
import sys
from pathlib import Path

import extract_env_vars as m

# ---------------------------------------------------------------------------
# find_python_files
# ---------------------------------------------------------------------------


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_find_python_files_excludes_tests_dir(tmp_path):
    _write(tmp_path / "app" / "agent.py")
    _write(tmp_path / "app" / "tools.py")
    _write(tmp_path / "tests" / "test_agent.py")
    _write(tmp_path / "app" / "tests" / "test_nested.py")
    _write(tmp_path / "README.md", "# not python")

    found = m.find_python_files(tmp_path)
    names = {p.name for p in found}

    assert names == {"agent.py", "tools.py"}
    # Result is sorted for determinism.
    assert found == sorted(found)


def test_find_python_files_skips_venv_and_cache_dirs(tmp_path):
    # Regression: previously find_python_files only skipped tests/, so a
    # local `.venv/`, `__pycache__/`, and various tool caches would be
    # scanned — polluting .env.example with vars from third-party packages
    # and matching hundreds of hardcoded model strings inside installed libs.
    _write(tmp_path / "app" / "agent.py")
    # Every directory that should now be skipped.
    for skipped in (
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".eggs",
        "build",
        "dist",
        ".git",
        "node_modules",
    ):
        _write(tmp_path / skipped / "junk.py", "MODEL = 'gemini-2.5-flash'\n")
        _write(
            tmp_path / skipped / "sub" / "more.py",
            "X = os.getenv('SHOULD_NOT_APPEAR')\n",
        )

    found = m.find_python_files(tmp_path)
    names = {p.name for p in found}

    # Only the recipe's own source file is returned.
    assert names == {"agent.py"}
    # Cross-check: no returned path contains any skipped dir in its parts.
    for p in found:
        assert "junk.py" != p.name
        assert "more.py" != p.name


# ---------------------------------------------------------------------------
# extract_env_vars / _extract_var_from_node
# ---------------------------------------------------------------------------


def test_extract_env_vars_all_forms(tmp_path):
    src = (
        "import os\n"
        "a = os.getenv('GETENV_PLAIN')\n"
        "b = os.getenv('GETENV_DEFAULT', 'dflt')\n"
        "c = os.environ.get('ENVIRON_GET')\n"
        "d = os.environ.get('ENVIRON_GET_DEFAULT', 'x')\n"
        "e = os.environ['ENVIRON_SUBSCRIPT']\n"
    )
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert result == {
        "GETENV_PLAIN": None,
        "GETENV_DEFAULT": "dflt",
        "ENVIRON_GET": None,
        "ENVIRON_GET_DEFAULT": "x",
        "ENVIRON_SUBSCRIPT": None,
    }


def test_extract_env_vars_ignores_non_uppercase_names(tmp_path):
    src = "import os\nx = os.getenv('lower_case')\ny = os.getenv('OK_NAME')\n"
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert result == {"OK_NAME": None}


def test_extract_env_vars_warns_on_non_uppercase_names(tmp_path, capsys):
    # Issue 3: lowercase names are dropped, but the drop must be surfaced on
    # stderr so the maintainer isn't left wondering why the var vanished.
    src = "import os\nx = os.getenv('lower_case')\ny = os.getenv('OK_NAME')\n"
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert result == {"OK_NAME": None}
    err = capsys.readouterr().err
    assert "lower_case" in err
    assert "UPPER_SNAKE_CASE" in err


def test_extract_env_vars_non_none_default_wins(tmp_path):
    # Same var appears first without a default, then with one.
    src = "import os\na = os.getenv('DUP')\nb = os.getenv('DUP', 'winner')\n"
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert result == {"DUP": "winner"}


def test_extract_env_vars_skips_unparseable_file(tmp_path, capsys):
    bad = _write(tmp_path / "bad.py", "def broken(:\n")
    good = _write(tmp_path / "good.py", "import os\nx = os.getenv('GOOD')\n")

    result = m.extract_env_vars([bad, good])

    assert result == {"GOOD": None}
    assert "Could not parse" in capsys.readouterr().err


def test_extract_env_vars_ignores_non_string_defaults(tmp_path):
    # Finding 4: a non-string literal default (e.g. an int) cannot be an env
    # value, so the default resolves to None rather than raising.
    src = (
        "import os\nP = os.getenv('PORT', 8080)\nB = os.getenv('FLAG', True)\n"
    )
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert result == {"PORT": None, "FLAG": None}


def test_extract_env_vars_ignores_binary_files(tmp_path):
    # Finding 6: files that raise UnicodeDecodeError are skipped, not fatal.
    binary = tmp_path / "blob.py"
    binary.write_bytes(b'\xff\xfe\x00 not utf-8 os.getenv("X")')
    good = _write(tmp_path / "good.py", "import os\nx = os.getenv('GOOD')\n")

    result = m.extract_env_vars([binary, good])

    assert result == {"GOOD": None}


# ---------------------------------------------------------------------------
# read_defined_vars
# ---------------------------------------------------------------------------


def test_read_defined_vars_missing_file(tmp_path):
    assert m.read_defined_vars(tmp_path / "nope.env") == set()


def test_read_defined_vars_parses_and_handles_export_and_comments(tmp_path):
    env = _write(
        tmp_path / ".env.example",
        "# a comment\n"
        "\n"
        "PLAIN=1\n"
        "export EXPORTED=2\n"
        "  SPACED = 3\n"
        "not a var line\n",
    )

    assert m.read_defined_vars(env) == {"PLAIN", "EXPORTED", "SPACED"}


# ---------------------------------------------------------------------------
# update_env_example
# ---------------------------------------------------------------------------


def test_update_env_example_creates_file_when_absent(tmp_path):
    env = tmp_path / ".env.example"

    added = m.update_env_example(env, {"NEW_VAR": "val", "NO_DEFAULT": None})

    assert added == ["NEW_VAR", "NO_DEFAULT"]
    content = env.read_text(encoding="utf-8")
    # Hard rule: every entry gets PLACEHOLDER, even when the source-code
    # default ("val" here) is passed in. See update_env_example's inline
    # comment for the rationale.
    assert f"NEW_VAR={m.PLACEHOLDER}" in content
    assert f"NO_DEFAULT={m.PLACEHOLDER}" in content
    assert "NEW_VAR=val" not in content


def test_update_env_example_appends_only_missing(tmp_path):
    env = _write(tmp_path / ".env.example", "EXISTING=1\n")

    added = m.update_env_example(env, {"EXISTING": None, "FRESH": "2"})

    assert added == ["FRESH"]
    content = env.read_text(encoding="utf-8")
    assert content.count("EXISTING") == 1
    # Passed-in "2" is ignored; every entry gets PLACEHOLDER.
    assert f"FRESH={m.PLACEHOLDER}" in content
    assert "FRESH=2" not in content


def test_update_env_example_noop_when_all_present(tmp_path):
    env = _write(tmp_path / ".env.example", "A=1\nB=2\n")
    before = env.read_text(encoding="utf-8")

    added = m.update_env_example(env, {"A": None, "B": None})

    assert added == []
    assert env.read_text(encoding="utf-8") == before


def test_update_env_example_handles_missing_trailing_newline(tmp_path):
    # No trailing newline on the existing content must not merge lines.
    env = _write(tmp_path / ".env.example", "A=1")

    m.update_env_example(env, {"B": "2"})

    lines = env.read_text(encoding="utf-8").splitlines()
    assert "A=1" in lines
    # Passed-in "2" is ignored; every entry gets PLACEHOLDER.
    assert f"B={m.PLACEHOLDER}" in lines
    assert "B=2" not in lines


def test_update_env_example_dry_run_reports_but_does_not_write(tmp_path):
    env = tmp_path / ".env.example"

    added = m.update_env_example(env, {"NEW": "1"}, dry_run=True)

    # Reports what it would add ...
    assert added == ["NEW"]
    # ... but writes nothing (file not even created).
    assert not env.exists()


# ---------------------------------------------------------------------------
# find_package_init
# ---------------------------------------------------------------------------


def test_find_package_init_returns_first_package(tmp_path):
    _write(tmp_path / "my_pkg" / "__init__.py", "")
    _write(tmp_path / "my_pkg" / "agent.py", "")

    init = m.find_package_init(tmp_path)

    assert init == tmp_path / "my_pkg" / "__init__.py"


def test_find_package_init_none_when_absent(tmp_path):
    _write(tmp_path / "plain_dir" / "file.py", "")

    assert m.find_package_init(tmp_path) is None


def test_find_package_init_excludes_tests_directory(tmp_path):
    # Finding 1: a tests/ package that sorts before the real package must not
    # be selected (it would receive the load_dotenv bootstrap by mistake).
    _write(tmp_path / "tests" / "__init__.py", "")
    _write(tmp_path / "zzz_agent" / "__init__.py", "")

    init = m.find_package_init(tmp_path)

    assert init == tmp_path / "zzz_agent" / "__init__.py"


def test_find_package_init_excludes_hidden_directory(tmp_path):
    _write(tmp_path / ".hidden_pkg" / "__init__.py", "")
    _write(tmp_path / "real_pkg" / "__init__.py", "")

    init = m.find_package_init(tmp_path)

    assert init == tmp_path / "real_pkg" / "__init__.py"


# ---------------------------------------------------------------------------
# inject_load_dotenv
# ---------------------------------------------------------------------------


def test_inject_load_dotenv_noop_if_present(tmp_path):
    init = _write(
        tmp_path / "__init__.py",
        "from dotenv import load_dotenv\nload_dotenv()\n",
    )
    before = init.read_text(encoding="utf-8")

    # Nothing to inject and no trailing relative imports → (False, 0).
    assert m.inject_load_dotenv(init) == (False, 0)
    assert init.read_text(encoding="utf-8") == before


def test_inject_load_dotenv_after_absolute_import_before_relative(tmp_path):
    init = _write(
        tmp_path / "__init__.py",
        '"""Package."""\n\nimport os\n\nfrom .agent import root_agent\n',
    )

    injected, noqa_added = m.inject_load_dotenv(init)
    assert injected is True
    # The trailing `from .agent` shifts below the injected load_dotenv() call,
    # so it also picks up a noqa: E402 marker.
    assert noqa_added == 1

    content = init.read_text(encoding="utf-8")
    ast.parse(content)  # result must be valid Python
    assert m.LOAD_DOTENV_IMPORT in content
    # load_dotenv must land AFTER the absolute import ...
    assert content.index("import os") < content.index("load_dotenv")
    # ... and BEFORE the relative import (env must be ready first).
    assert content.index("load_dotenv") < content.index("from .agent")


def test_inject_load_dotenv_no_imports_goes_after_docstring(tmp_path):
    init = _write(tmp_path / "__init__.py", '"""Package docstring."""\n')

    injected, noqa_added = m.inject_load_dotenv(init)
    assert injected is True
    assert noqa_added == 0  # no trailing relative imports at all

    content = init.read_text(encoding="utf-8")
    ast.parse(content)  # result must be valid Python
    assert content.index('"""Package docstring."""') < content.index(
        "load_dotenv"
    )


def test_inject_load_dotenv_adds_noqa_to_trailing_relative_imports(tmp_path):
    # Since load_dotenv() is a bare statement, Ruff would flag any relative
    # imports below it as E402. The injection must add a suppression comment
    # so the resulting file stays lint-clean without human edits.
    init = _write(
        tmp_path / "__init__.py",
        "import os\nfrom .agent import root_agent\n",
    )

    injected, noqa_added = m.inject_load_dotenv(init)
    assert injected is True
    assert noqa_added == 1

    content = init.read_text(encoding="utf-8")
    ast.parse(content)  # result must be valid Python
    rel_line = next(
        ln for ln in content.splitlines() if ln.startswith("from .agent")
    )
    assert "noqa: E402" in rel_line
    # Idempotent: a second run must not duplicate the suffix.
    assert m.inject_load_dotenv(init) == (False, 0)
    assert init.read_text(encoding="utf-8").count("noqa: E402") == 1


def test_inject_load_dotenv_ignores_docstring_mention(tmp_path):
    # A mention of load_dotenv in a docstring/comment must NOT suppress a real
    # injection (AST-based idempotency check, not a fragile substring search).
    init = _write(
        tmp_path / "__init__.py",
        '"""We will load_dotenv somewhere."""\nimport os\n',
    )

    injected, noqa_added = m.inject_load_dotenv(init)
    assert injected is True
    assert noqa_added == 0  # no trailing relative imports

    content = init.read_text(encoding="utf-8")
    ast.parse(content)  # result must be valid Python
    assert m.LOAD_DOTENV_IMPORT in content


def test_inject_load_dotenv_dry_run_reports_but_does_not_write(tmp_path):
    original = "import os\n"
    init = _write(tmp_path / "__init__.py", original)

    # Reports that it would inject ...
    injected, noqa_added = m.inject_load_dotenv(init, dry_run=True)
    assert injected is True
    assert noqa_added == 0
    # ... but leaves the file untouched.
    assert init.read_text(encoding="utf-8") == original


def test_inject_load_dotenv_not_placed_inside_docstring(tmp_path):
    # Regression (Issue 1, failure mode A): an ``import`` line inside the
    # module docstring must NOT be mistaken for a real import. The old text
    # scanner inserted the bootstrap inside the triple-quoted string, where
    # load_dotenv() silently never executed.
    init = _write(
        tmp_path / "__init__.py",
        '"""\n'
        "Usage example::\n"
        "    import os\n"
        '    KEY = os.getenv("API_KEY")\n'
        '"""\n'
        "from .agent import root_agent\n",
    )

    injected, noqa_added = m.inject_load_dotenv(init)
    assert injected is True
    # The pre-existing `from .agent import root_agent` shifts below the
    # injected load_dotenv() call, so it picks up a noqa: E402 marker.
    assert noqa_added == 1

    content = init.read_text(encoding="utf-8")
    tree = ast.parse(content)  # (b) result is valid Python
    # (a) load_dotenv() is a real top-level statement, not text in a string.
    top_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "load_dotenv"
    ]
    assert len(top_level_calls) == 1
    # (c) the docstring value is unchanged (still contains the example import).
    docstring = ast.get_docstring(tree)
    assert docstring is not None
    assert "import os" in docstring


def test_inject_load_dotenv_ignores_import_inside_conditional(tmp_path):
    # Regression (Issue 1, failure mode B): an import nested in an if-block is
    # not a top-level import. The old text scanner inserted the bootstrap
    # after it, dedenting into the block and raising IndentationError.
    init = _write(
        tmp_path / "__init__.py",
        "import os\nif True:\n    import pdb\n    x = 1\n",
    )

    injected, noqa_added = m.inject_load_dotenv(init)
    assert injected is True
    assert noqa_added == 0  # no trailing relative imports

    content = init.read_text(encoding="utf-8")
    ast.parse(content)  # must remain valid Python (no IndentationError)
    # load_dotenv lands after the real top-level import, before the block.
    assert content.index("import os") < content.index("load_dotenv")
    assert content.index("load_dotenv") < content.index("if True:")


def test_inject_load_dotenv_marks_late_relative_import_when_already_bootstrapped(
    tmp_path,
):
    # Regression: some recipe authors hand-write the env-bootstrap pattern
    # (load_dotenv + os.environ.setdefault + trailing `from .agent`) without
    # a noqa marker on the relative import. When we run against such a file
    # we must NOT inject anything (load_dotenv is already there) but MUST
    # still add `# noqa: E402` to the trailing relative import — otherwise
    # ruff (Phase 4 in prepare-python-recipe) flags E402 for a pattern the
    # skill is aware of and could have suppressed.
    init = _write(
        tmp_path / "__init__.py",
        "import os\n"
        "from dotenv import load_dotenv\n"
        "\n"
        "load_dotenv()\n"
        "os.environ.setdefault('FOO', 'bar')\n"
        "\n"
        "from . import agent\n",
    )

    injected, noqa_added = m.inject_load_dotenv(init)
    assert injected is False  # load_dotenv was already present
    assert noqa_added == 1  # ... but the trailing relative import was marked

    content = init.read_text(encoding="utf-8")
    rel_line = next(
        ln for ln in content.splitlines() if ln.startswith("from . import")
    )
    assert "noqa: E402" in rel_line

    # Idempotent on a second run — nothing more to do.
    assert m.inject_load_dotenv(init) == (False, 0)
    assert init.read_text(encoding="utf-8").count("noqa: E402") == 1


def test_inject_load_dotenv_leaves_early_relative_import_alone(tmp_path):
    # Precision check: a relative import at the TOP of __init__.py (before
    # any non-import statement) does NOT trigger Ruff E402. The suppression
    # pass must be precise — the previous implementation blindly marked
    # every relative import, which produced meaningless noqa comments on
    # perfectly-fine lines.
    init = _write(
        tmp_path / "__init__.py",
        "from . import agent\n"
        "from dotenv import load_dotenv\n"
        "\n"
        "load_dotenv()\n",
    )

    injected, noqa_added = m.inject_load_dotenv(init)
    assert injected is False
    assert noqa_added == 0

    content = init.read_text(encoding="utf-8")
    rel_line = next(
        ln for ln in content.splitlines() if ln.startswith("from . import")
    )
    assert "noqa" not in rel_line


def test_inject_load_dotenv_ignores_docstring_before_late_relative_import(
    tmp_path,
):
    # A module docstring at the top of the file must NOT count as "the first
    # non-import statement" — otherwise every relative import in a module with
    # a docstring would be treated as late and get spuriously marked.
    init = _write(
        tmp_path / "__init__.py",
        '"""My package."""\n'
        "\n"
        "from dotenv import load_dotenv\n"
        "\n"
        "load_dotenv()\n"
        "\n"
        "from . import agent\n",
    )

    injected, noqa_added = m.inject_load_dotenv(init)
    assert injected is False  # already bootstrapped
    # The trailing `from . import agent` DOES come after load_dotenv() (not
    # after the docstring), so it IS late and should be marked.
    assert noqa_added == 1

    content = init.read_text(encoding="utf-8")
    rel_line = next(
        ln for ln in content.splitlines() if ln.startswith("from . import")
    )
    assert "noqa: E402" in rel_line


# ---------------------------------------------------------------------------
# ensure_python_dotenv_dependency
# ---------------------------------------------------------------------------


def test_ensure_dotenv_missing_file(tmp_path):
    assert m.ensure_python_dotenv_dependency(tmp_path / "nope.toml") is False


def test_ensure_dotenv_noop_when_in_project_deps(tmp_path):
    pyproject = _write(
        tmp_path / "pyproject.toml",
        '[project]\ndependencies = [\n    "python-dotenv>=1.0.0",\n]\n',
    )
    before = pyproject.read_text(encoding="utf-8")

    assert m.ensure_python_dotenv_dependency(pyproject) is False
    assert pyproject.read_text(encoding="utf-8") == before


def test_ensure_dotenv_multiline_array(tmp_path):
    pyproject = _write(
        tmp_path / "pyproject.toml",
        '[project]\ndependencies = [\n    "requests",\n]\n',
    )

    assert m.ensure_python_dotenv_dependency(pyproject) is True

    content = pyproject.read_text(encoding="utf-8")
    assert '"python-dotenv>=1.0.0"' in content
    assert '"requests"' in content


def test_ensure_dotenv_single_line_array_gets_comma(tmp_path):
    pyproject = _write(
        tmp_path / "pyproject.toml",
        '[project]\ndependencies = ["requests"]\n',
    )

    assert m.ensure_python_dotenv_dependency(pyproject) is True

    content = pyproject.read_text(encoding="utf-8")
    assert '"requests",' in content
    assert '"python-dotenv>=1.0.0"' in content


def test_ensure_dotenv_added_when_only_in_dev_group(tmp_path):
    # python-dotenv present in a dependency group but NOT in [project] deps:
    # it must still be added to the main dependencies.
    pyproject = _write(
        tmp_path / "pyproject.toml",
        "[project]\n"
        'dependencies = [\n    "requests",\n]\n\n'
        "[dependency-groups]\n"
        'dev = ["python-dotenv>=1.0.0"]\n',
    )

    assert m.ensure_python_dotenv_dependency(pyproject) is True

    content = pyproject.read_text(encoding="utf-8")
    # Now present in the [project] block, not only the group.
    project_block = content.split("[dependency-groups]")[0]
    assert "python-dotenv" in project_block


def test_ensure_dotenv_creates_dependencies_block_if_missing(tmp_path):
    # Finding 2: a [project] table with no dependencies array at all must get
    # one created rather than being silently skipped.
    pyproject = _write(
        tmp_path / "pyproject.toml",
        '[project]\nname = "x"\nversion = "0.1.0"\n',
    )

    assert m.ensure_python_dotenv_dependency(pyproject) is True

    content = pyproject.read_text(encoding="utf-8")
    assert "dependencies = [" in content
    assert '"python-dotenv>=1.0.0"' in content
    # The result must be valid TOML with dotenv under [project].dependencies.
    import tomllib

    data = tomllib.loads(content)
    assert "python-dotenv>=1.0.0" in data["project"]["dependencies"]


def test_ensure_dotenv_noop_when_present_alongside_extras(tmp_path):
    # Regression: the previous regex-based detector used non-greedy `.*?\]`,
    # so the FIRST `]` (inside `google-adk[gcp]`) closed the match early.
    # Result: the detector missed later entries including `python-dotenv` and
    # tried to add it again, mangling the file. The parser-based detector
    # must correctly see `python-dotenv` and no-op.
    pyproject = _write(
        tmp_path / "pyproject.toml",
        "[project]\n"
        "dependencies = [\n"
        '    "google-adk[gcp]>=2.0.0,<3.0.0",\n'
        '    "python-dotenv>=1.0.0",\n'
        "]\n",
    )
    before = pyproject.read_text(encoding="utf-8")

    assert m.ensure_python_dotenv_dependency(pyproject) is False
    assert pyproject.read_text(encoding="utf-8") == before


def test_ensure_dotenv_insert_preserves_extras_and_valid_toml(tmp_path):
    # Regression: the previous regex-based inserter used non-greedy `.*?\]`,
    # so it treated the `]` inside `google-adk[gcp]` as the array's closing
    # bracket and injected `"python-dotenv"` INSIDE the extras, producing
    # syntactically broken TOML. The bracket-depth-aware inserter must place
    # `python-dotenv` outside all extras, and the result must round-trip
    # through tomllib.
    import tomllib

    pyproject = _write(
        tmp_path / "pyproject.toml",
        "[project]\n"
        "dependencies = [\n"
        '    "google-adk[gcp]>=2.0.0,<3.0.0",\n'
        '    "requests",\n'
        "]\n",
    )

    assert m.ensure_python_dotenv_dependency(pyproject) is True

    content = pyproject.read_text(encoding="utf-8")
    # Extras remain intact.
    assert '"google-adk[gcp]>=2.0.0,<3.0.0"' in content
    # dotenv landed.
    assert '"python-dotenv>=1.0.0"' in content
    # The file is valid TOML and lists python-dotenv under [project].deps.
    data = tomllib.loads(content)
    assert "python-dotenv>=1.0.0" in data["project"]["dependencies"]
    assert "google-adk[gcp]>=2.0.0,<3.0.0" in data["project"]["dependencies"]


def test_ensure_dotenv_refuses_to_write_invalid_toml(tmp_path, monkeypatch):
    # Belt-and-braces: even if the low-level inserter had a bug, the wrapper
    # must round-trip the result through tomllib and refuse to write output
    # that isn't valid TOML with python-dotenv in [project].dependencies.
    pyproject = _write(
        tmp_path / "pyproject.toml",
        '[project]\ndependencies = [\n    "requests",\n]\n',
    )
    original = pyproject.read_text(encoding="utf-8")

    # Force the insertion helper to return garbage.
    monkeypatch.setattr(m, "_insert_before_close", lambda c, i: "!!!not toml")

    assert m.ensure_python_dotenv_dependency(pyproject) is False
    # File is unchanged on refusal.
    assert pyproject.read_text(encoding="utf-8") == original


def test_ensure_dotenv_no_project_table_returns_false(tmp_path):
    # Nothing sensible to modify when there is no [project] table.
    pyproject = _write(tmp_path / "pyproject.toml", "[tool.foo]\nx = 1\n")

    assert m.ensure_python_dotenv_dependency(pyproject) is False
    assert "python-dotenv" not in pyproject.read_text(encoding="utf-8")


def test_ensure_dotenv_dry_run_reports_but_does_not_write(tmp_path):
    original = '[project]\ndependencies = [\n    "requests",\n]\n'
    pyproject = _write(tmp_path / "pyproject.toml", original)

    # Reports that it would add the dependency ...
    assert m.ensure_python_dotenv_dependency(pyproject, dry_run=True) is True
    # ... but leaves pyproject.toml untouched.
    assert pyproject.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# _model_str_to_suffix / assign_model_var_names
# ---------------------------------------------------------------------------


def test_model_str_to_suffix():
    assert m._model_str_to_suffix("gemini-3.5-flash") == "GEMINI_3_5_FLASH"
    assert (
        m._model_str_to_suffix("gemini-embedding-001") == "GEMINI_EMBEDDING_001"
    )
    assert m._model_str_to_suffix("claude-3-sonnet") == "CLAUDE_3_SONNET"


def test_assign_model_var_names_single():
    assert m.assign_model_var_names({"gemini-3.5-flash"}) == {
        "gemini-3.5-flash": "MODEL_NAME"
    }


def test_assign_model_var_names_multiple():
    result = m.assign_model_var_names({"gemini-3.5-flash", "claude-3-sonnet"})
    assert result == {
        "gemini-3.5-flash": "MODEL_NAME_GEMINI_3_5_FLASH",
        "claude-3-sonnet": "MODEL_NAME_CLAUDE_3_SONNET",
    }


def test_assign_model_var_names_collision_disambiguated():
    # Both strings normalise to the same suffix; sorted order decides which
    # keeps the base name and which gets the _2 suffix.
    result = m.assign_model_var_names({"gemini-3.5-flash", "gemini-3-5-flash"})
    assert result["gemini-3-5-flash"] == "MODEL_NAME_GEMINI_3_5_FLASH"
    assert result["gemini-3.5-flash"] == "MODEL_NAME_GEMINI_3_5_FLASH_2"


# ---------------------------------------------------------------------------
# extract_hardcoded_models
# ---------------------------------------------------------------------------


def test_extract_hardcoded_models_detects_and_skips_docstrings(tmp_path):
    src = (
        '"""This mentions gemini-3.5-flash in a docstring."""\n'
        "import os\n"
        'MODEL = "gemini-3.5-flash"\n'
        'OTHER = "not-a-model"\n'
    )
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])

    assert py in hits
    found_strings = [s for _lineno, s in hits[py]]
    assert found_strings == ["gemini-3.5-flash"]  # docstring not included


# ---------------------------------------------------------------------------
# replace_hardcoded_models
# ---------------------------------------------------------------------------


def test_replace_hardcoded_models_replaces_and_adds_import_os(tmp_path):
    src = 'MODEL = "gemini-3.5-flash"\n'
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    name_map = m.assign_model_var_names({"gemini-3.5-flash"})
    substituted = m.replace_hardcoded_models([py], hits, name_map)

    assert substituted == {"gemini-3.5-flash": "MODEL_NAME"}
    content = py.read_text(encoding="utf-8")
    ast.parse(content)  # result must be valid Python
    assert 'os.getenv("MODEL_NAME")' in content
    assert "import os" in content
    assert "gemini-3.5-flash" not in content


def test_replace_hardcoded_models_handles_single_quotes(tmp_path):
    # AST-position based replacement must handle any quote style.
    src = "import os\nMODEL = 'gemini-3.5-flash'\n"
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    name_map = m.assign_model_var_names({"gemini-3.5-flash"})
    m.replace_hardcoded_models([py], hits, name_map)

    content = py.read_text(encoding="utf-8")
    ast.parse(content)  # result must be valid Python
    assert 'os.getenv("MODEL_NAME")' in content
    assert "gemini-3.5-flash" not in content


def test_replace_hardcoded_models_injects_os_even_if_substring_in_docstring(
    tmp_path,
):
    # Finding 3: "import os" present only inside a docstring must not fool the
    # import check — a real `import os` statement must still be added, otherwise
    # the injected os.getenv(...) would raise NameError at runtime.
    src = (
        '"""Example that says import os in prose."""\n'
        'MODEL = "gemini-3.5-flash"\n'
    )
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    name_map = m.assign_model_var_names({"gemini-3.5-flash"})
    m.replace_hardcoded_models([py], hits, name_map)

    content = py.read_text(encoding="utf-8")
    # A real, top-level `import os` statement was added ...
    assert any(line.strip() == "import os" for line in content.splitlines())
    # ... and the resulting module is syntactically valid.
    ast.parse(content)


def test_replace_hardcoded_models_multiple_occurrences(tmp_path):
    # Finding 5: multiple replacements (incl. a duplicate) must all apply
    # without offset drift, thanks to reverse-order substitution.
    src = (
        "import os\n"
        'M1 = "gemini-3.5-flash"\n'
        'M2 = "claude-3-sonnet"\n'
        'M3 = "gemini-3.5-flash"\n'
    )
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    strings = {s for file_hits in hits.values() for _lineno, s in file_hits}
    name_map = m.assign_model_var_names(strings)
    m.replace_hardcoded_models([py], hits, name_map)

    content = py.read_text(encoding="utf-8")
    assert "gemini-3.5-flash" not in content
    assert "claude-3-sonnet" not in content
    # Both duplicate occurrences map to the same env var.
    assert content.count('os.getenv("MODEL_NAME_GEMINI_3_5_FLASH")') == 2
    assert content.count('os.getenv("MODEL_NAME_CLAUDE_3_SONNET")') == 1
    ast.parse(content)


def test_replace_hardcoded_models_dry_run_reports_but_does_not_write(tmp_path):
    original = 'MODEL = "gemini-3.5-flash"\n'
    py = _write(tmp_path / "mod.py", original)

    hits = m.extract_hardcoded_models([py])
    name_map = m.assign_model_var_names({"gemini-3.5-flash"})
    substituted = m.replace_hardcoded_models([py], hits, name_map, dry_run=True)

    # Reports the substitution it would make ...
    assert substituted == {"gemini-3.5-flash": "MODEL_NAME"}
    # ... but leaves the source file untouched.
    assert py.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# helpers: _post_header_index / _docstring_node_ids / _flat_offset
# ---------------------------------------------------------------------------


def test_post_header_index_after_comments_and_docstring():
    lines = [
        "# license\n",
        "# more\n",
        "\n",
        '"""Doc."""\n',
        "import os\n",
    ]
    assert m._post_header_index(lines) == 4


def test_post_header_index_multiline_docstring():
    lines = [
        '"""First line\n',
        "second line\n",
        'closing."""\n',
        "import os\n",
    ]
    assert m._post_header_index(lines) == 3


def test_post_header_index_no_header():
    lines = ["import os\n"]
    assert m._post_header_index(lines) == 0


def test_docstring_node_ids_marks_module_docstring():
    src = '"""Module doc."""\nx = "not a docstring"\n'
    tree = ast.parse(src)
    ids = m._docstring_node_ids(tree)

    module_doc = tree.body[0].value  # the docstring Constant
    assert id(module_doc) in ids


def test_flat_offset():
    lines = ["abc\n", "defg\n"]
    # Start of line 2 (1-based), col 0 → offset past "abc\n" == 4
    assert m._flat_offset(lines, 2, 0) == 4
    # Line 2, col 2 → 4 + 2 == 6
    assert m._flat_offset(lines, 2, 2) == 6


# ---------------------------------------------------------------------------
# run_step_pyproject messaging (Issue 4)
# ---------------------------------------------------------------------------


def test_run_step_pyproject_warns_when_insertion_not_possible(tmp_path, capsys):
    # Issue 4: python-dotenv is genuinely absent and cannot be inserted (no
    # [project] table). The step must WARN "add by hand" rather than falsely
    # printing "[PASS] already includes".
    _write(tmp_path / "pyproject.toml", "[tool.foo]\nx = 1\n")

    m.run_step_pyproject(tmp_path)

    out = capsys.readouterr().out
    assert "[WARN] Could not safely add python-dotenv" in out
    assert "already includes" not in out


def test_run_step_pyproject_pass_when_already_present(tmp_path, capsys):
    _write(
        tmp_path / "pyproject.toml",
        '[project]\ndependencies = [\n    "python-dotenv>=1.0.0",\n]\n',
    )

    m.run_step_pyproject(tmp_path)

    out = capsys.readouterr().out
    assert "[PASS] pyproject.toml already includes python-dotenv" in out


# ---------------------------------------------------------------------------
# main() end-to-end: --dry-run vs a real run
# ---------------------------------------------------------------------------


def _make_sample_recipe(root: Path) -> Path:
    """A minimal recipe: a package that reads an env var and hardcodes a
    model, plus a pyproject.toml without python-dotenv and no .env.example."""
    recipe = root / "my-recipe"
    pkg = recipe / "my_recipe"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "agent.py").write_text(
        "import os\n"
        'KEY = os.getenv("MY_API_KEY")\n'
        'MODEL = "gemini-3.5-flash"\n',
        encoding="utf-8",
    )
    (recipe / "pyproject.toml").write_text(
        '[project]\nname = "my-recipe"\ndependencies = [\n    "requests",\n]\n',
        encoding="utf-8",
    )
    return recipe


def _run_main(monkeypatch, recipe: Path, *extra_args: str) -> None:
    argv = ["extract_env_vars", "--recipe-dir", str(recipe), *extra_args]
    monkeypatch.setattr(sys, "argv", argv)
    m.main()


def test_main_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    recipe = _make_sample_recipe(tmp_path)
    init_py = recipe / "my_recipe" / "__init__.py"
    agent_py = recipe / "my_recipe" / "agent.py"
    pyproject = recipe / "pyproject.toml"
    before = {
        p: p.read_text(encoding="utf-8") for p in (init_py, agent_py, pyproject)
    }

    _run_main(monkeypatch, recipe, "--dry-run")

    # No file was created or modified.
    assert not (recipe / ".env.example").exists()
    for p, text in before.items():
        assert p.read_text(encoding="utf-8") == text

    out = capsys.readouterr().out
    assert "[DRY-RUN]" in out
    assert "no files will be modified" in out


def test_main_real_run_applies_changes(tmp_path, monkeypatch):
    recipe = _make_sample_recipe(tmp_path)
    init_py = recipe / "my_recipe" / "__init__.py"
    agent_py = recipe / "my_recipe" / "agent.py"
    pyproject = recipe / "pyproject.toml"

    _run_main(monkeypatch, recipe)

    # .env.example created with the detected env var and the extracted model.
    env_text = (recipe / ".env.example").read_text(encoding="utf-8")
    assert "MY_API_KEY=" in env_text
    assert "MODEL_NAME=" in env_text
    # load_dotenv injected, python-dotenv added, model string replaced.
    assert "load_dotenv" in init_py.read_text(encoding="utf-8")
    assert "python-dotenv" in pyproject.read_text(encoding="utf-8")
    agent_text = agent_py.read_text(encoding="utf-8")
    # Hard rule: the replacement is BARE os.getenv("MODEL_NAME") — no inferred
    # default is written, even though the original model string is known.
    assert 'os.getenv("MODEL_NAME")' in agent_text
    assert 'os.getenv("MODEL_NAME", ' not in agent_text
    assert "gemini-3.5-flash" not in agent_text
