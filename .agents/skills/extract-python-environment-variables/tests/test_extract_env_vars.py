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
    assert "NEW_VAR=val" in content
    assert f"NO_DEFAULT={m.PLACEHOLDER}" in content


def test_update_env_example_appends_only_missing(tmp_path):
    env = _write(tmp_path / ".env.example", "EXISTING=1\n")

    added = m.update_env_example(env, {"EXISTING": None, "FRESH": "2"})

    assert added == ["FRESH"]
    content = env.read_text(encoding="utf-8")
    assert content.count("EXISTING") == 1
    assert "FRESH=2" in content


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
    assert "B=2" in lines


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

    assert m.inject_load_dotenv(init) is False
    assert init.read_text(encoding="utf-8") == before


def test_inject_load_dotenv_after_absolute_import_before_relative(tmp_path):
    init = _write(
        tmp_path / "__init__.py",
        '"""Package."""\n\nimport os\n\nfrom .agent import root_agent\n',
    )

    assert m.inject_load_dotenv(init) is True

    content = init.read_text(encoding="utf-8")
    assert m.LOAD_DOTENV_IMPORT in content
    # load_dotenv must land AFTER the absolute import ...
    assert content.index("import os") < content.index("load_dotenv")
    # ... and BEFORE the relative import (env must be ready first).
    assert content.index("load_dotenv") < content.index("from .agent")


def test_inject_load_dotenv_no_imports_goes_after_docstring(tmp_path):
    init = _write(tmp_path / "__init__.py", '"""Package docstring."""\n')

    assert m.inject_load_dotenv(init) is True

    content = init.read_text(encoding="utf-8")
    assert content.index('"""Package docstring."""') < content.index(
        "load_dotenv"
    )


def test_inject_load_dotenv_ignores_docstring_mention(tmp_path):
    # A mention of load_dotenv in a docstring/comment must NOT suppress a real
    # injection (AST-based idempotency check, not a fragile substring search).
    init = _write(
        tmp_path / "__init__.py",
        '"""We will load_dotenv somewhere."""\nimport os\n',
    )

    assert m.inject_load_dotenv(init) is True

    content = init.read_text(encoding="utf-8")
    assert m.LOAD_DOTENV_IMPORT in content


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


def test_ensure_dotenv_no_project_table_returns_false(tmp_path):
    # Nothing sensible to modify when there is no [project] table.
    pyproject = _write(tmp_path / "pyproject.toml", "[tool.foo]\nx = 1\n")

    assert m.ensure_python_dotenv_dependency(pyproject) is False
    assert "python-dotenv" not in pyproject.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# _model_str_to_suffix / assign_model_var_names
# ---------------------------------------------------------------------------


def test_model_str_to_suffix():
    assert m._model_str_to_suffix("gemini-2.5-flash") == "GEMINI_2_5_FLASH"
    assert (
        m._model_str_to_suffix("gemini-embedding-001") == "GEMINI_EMBEDDING_001"
    )
    assert m._model_str_to_suffix("claude-3-sonnet") == "CLAUDE_3_SONNET"


def test_assign_model_var_names_single():
    assert m.assign_model_var_names({"gemini-2.5-flash"}) == {
        "gemini-2.5-flash": "MODEL_NAME"
    }


def test_assign_model_var_names_multiple():
    result = m.assign_model_var_names({"gemini-2.5-flash", "claude-3-sonnet"})
    assert result == {
        "gemini-2.5-flash": "MODEL_NAME_GEMINI_2_5_FLASH",
        "claude-3-sonnet": "MODEL_NAME_CLAUDE_3_SONNET",
    }


def test_assign_model_var_names_collision_disambiguated():
    # Both strings normalise to the same suffix; sorted order decides which
    # keeps the base name and which gets the _2 suffix.
    result = m.assign_model_var_names({"gemini-2.5-flash", "gemini-2-5-flash"})
    assert result["gemini-2-5-flash"] == "MODEL_NAME_GEMINI_2_5_FLASH"
    assert result["gemini-2.5-flash"] == "MODEL_NAME_GEMINI_2_5_FLASH_2"


# ---------------------------------------------------------------------------
# extract_hardcoded_models
# ---------------------------------------------------------------------------


def test_extract_hardcoded_models_detects_and_skips_docstrings(tmp_path):
    src = (
        '"""This mentions gemini-2.5-flash in a docstring."""\n'
        "import os\n"
        'MODEL = "gemini-2.5-flash"\n'
        'OTHER = "not-a-model"\n'
    )
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])

    assert py in hits
    found_strings = [s for _lineno, s in hits[py]]
    assert found_strings == ["gemini-2.5-flash"]  # docstring not included


# ---------------------------------------------------------------------------
# replace_hardcoded_models
# ---------------------------------------------------------------------------


def test_replace_hardcoded_models_replaces_and_adds_import_os(tmp_path):
    src = 'MODEL = "gemini-2.5-flash"\n'
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    name_map = m.assign_model_var_names({"gemini-2.5-flash"})
    substituted = m.replace_hardcoded_models([py], hits, name_map)

    assert substituted == {"gemini-2.5-flash": "MODEL_NAME"}
    content = py.read_text(encoding="utf-8")
    assert 'os.getenv("MODEL_NAME")' in content
    assert "import os" in content
    assert "gemini-2.5-flash" not in content


def test_replace_hardcoded_models_handles_single_quotes(tmp_path):
    # AST-position based replacement must handle any quote style.
    src = "import os\nMODEL = 'gemini-2.5-flash'\n"
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    name_map = m.assign_model_var_names({"gemini-2.5-flash"})
    m.replace_hardcoded_models([py], hits, name_map)

    content = py.read_text(encoding="utf-8")
    assert 'os.getenv("MODEL_NAME")' in content
    assert "gemini-2.5-flash" not in content


def test_replace_hardcoded_models_injects_os_even_if_substring_in_docstring(
    tmp_path,
):
    # Finding 3: "import os" present only inside a docstring must not fool the
    # import check — a real `import os` statement must still be added, otherwise
    # the injected os.getenv(...) would raise NameError at runtime.
    src = (
        '"""Example that says import os in prose."""\n'
        'MODEL = "gemini-2.5-flash"\n'
    )
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    name_map = m.assign_model_var_names({"gemini-2.5-flash"})
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
        'M1 = "gemini-2.5-flash"\n'
        'M2 = "claude-3-sonnet"\n'
        'M3 = "gemini-2.5-flash"\n'
    )
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    strings = {s for file_hits in hits.values() for _lineno, s in file_hits}
    name_map = m.assign_model_var_names(strings)
    m.replace_hardcoded_models([py], hits, name_map)

    content = py.read_text(encoding="utf-8")
    assert "gemini-2.5-flash" not in content
    assert "claude-3-sonnet" not in content
    # Both duplicate occurrences map to the same env var.
    assert content.count('os.getenv("MODEL_NAME_GEMINI_2_5_FLASH")') == 2
    assert content.count('os.getenv("MODEL_NAME_CLAUDE_3_SONNET")') == 1
    ast.parse(content)


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
