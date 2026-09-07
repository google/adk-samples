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


def test_find_python_files_skips_conftest_py_at_any_level(tmp_path):
    # REGRESSION: pytest's `conftest.py` is a reserved filename used only
    # for test collection/fixture setup — never for application code.
    # A common pattern is to put it at the recipe ROOT (not inside a
    # `tests/` dir) so it applies to the whole project. The tests/ dir
    # exclusion misses it there, and env vars in conftest.py (e.g.
    # `INTEGRATION_TEST` guarding integration-test collection) would leak
    # into .env.example — silently inviting users to set test-mode toggles.
    _write(tmp_path / "app" / "agent.py")
    _write(
        tmp_path / "conftest.py",
        "import os\nos.environ.get('INTEGRATION_TEST')\n",
    )
    _write(tmp_path / "app" / "conftest.py", "# also skipped\n")
    _write(
        tmp_path / "some_subpkg" / "conftest.py",
        "import os\nos.environ.get('ANOTHER_TEST_VAR')\n",
    )

    found = m.find_python_files(tmp_path)
    names = {p.name for p in found}

    # Only the recipe's actual application source is returned.
    assert names == {"agent.py"}
    # Belt-and-braces: no returned file is named conftest.py.
    assert not any(p.name == "conftest.py" for p in found)


def test_extract_env_vars_ignores_conftest_py_content(tmp_path):
    # End-to-end: env vars declared inside conftest.py never surface in
    # the aggregated extract_env_vars() result, because the scanner never
    # opens the file in the first place.
    _write(
        tmp_path / "conftest.py",
        "import os\nX = os.getenv('INTEGRATION_TEST')\n",
    )
    _write(
        tmp_path / "app.py",
        "import os\nY = os.getenv('REAL_APP_VAR', 'v')\n",
    )

    py_files = m.find_python_files(tmp_path)
    result = m.extract_env_vars(py_files)

    assert "INTEGRATION_TEST" not in result  # conftest content excluded
    assert "REAL_APP_VAR" in result  # app content still scanned


# ---------------------------------------------------------------------------
# extract_env_vars / _extract_var_from_node
# ---------------------------------------------------------------------------


# Helper to compress the new richer return type down to something tests
# can assert against concisely. Each var maps to a list of (kind, default)
# tuples — the fields that matter for correctness — dropping file/line
# which are always the same in these tests.
def _kd(result):
    """Return {var: [(kind, default), ...]} from an extract_env_vars result."""
    return {
        v: [(s.kind, s.default) for s in srcs] for v, srcs in result.items()
    }


def test_extract_env_vars_all_forms(tmp_path):
    src = (
        "import os\n"
        "a = os.getenv('GETENV_PLAIN')\n"
        "b = os.getenv('GETENV_DEFAULT', 'dflt')\n"
        "c = os.environ.get('ENVIRON_GET')\n"
        "d = os.environ.get('ENVIRON_GET_DEFAULT', 'x')\n"
        "e = os.environ['ENVIRON_SUBSCRIPT']\n"
        "os.environ.setdefault('SETDEFAULT_VAR', 'sd_default')\n"
    )
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert _kd(result) == {
        "GETENV_PLAIN": [("getenv", None)],
        "GETENV_DEFAULT": [("getenv", "dflt")],
        "ENVIRON_GET": [("environ_get", None)],
        "ENVIRON_GET_DEFAULT": [("environ_get", "x")],
        "ENVIRON_SUBSCRIPT": [("environ_sub", None)],
        "SETDEFAULT_VAR": [("setdefault", "sd_default")],
    }
    # Every Source records the file it came from.
    assert all(s.file == py for srcs in result.values() for s in srcs)


def test_extract_env_vars_ignores_non_uppercase_names(tmp_path):
    src = "import os\nx = os.getenv('lower_case')\ny = os.getenv('OK_NAME')\n"
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert _kd(result) == {"OK_NAME": [("getenv", None)]}


def test_extract_env_vars_warns_on_non_uppercase_names(tmp_path, capsys):
    # Issue 3: lowercase names are dropped, but the drop must be surfaced on
    # stderr so the maintainer isn't left wondering why the var vanished.
    src = "import os\nx = os.getenv('lower_case')\ny = os.getenv('OK_NAME')\n"
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert _kd(result) == {"OK_NAME": [("getenv", None)]}
    err = capsys.readouterr().err
    assert "lower_case" in err
    assert "UPPER_SNAKE_CASE" in err


def test_extract_env_vars_records_all_occurrences(tmp_path):
    # v2: extract_env_vars keeps every occurrence rather than collapsing to
    # a single (var, default). The resolver picks the winner separately.
    src = "import os\na = os.getenv('DUP')\nb = os.getenv('DUP', 'x')\n"
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    # Two sources for the same var: one bare, one with a default.
    assert len(result["DUP"]) == 2
    kinds_defaults = {(s.kind, s.default) for s in result["DUP"]}
    assert kinds_defaults == {("getenv", None), ("getenv", "x")}


def test_extract_env_vars_skips_unparseable_file(tmp_path, capsys):
    bad = _write(tmp_path / "bad.py", "def broken(:\n")
    good = _write(tmp_path / "good.py", "import os\nx = os.getenv('GOOD')\n")

    result = m.extract_env_vars([bad, good])

    assert _kd(result) == {"GOOD": [("getenv", None)]}
    assert "Could not parse" in capsys.readouterr().err


def test_extract_env_vars_ignores_non_string_defaults(tmp_path):
    # Finding 4: a non-string literal default (e.g. an int) cannot be an env
    # value, so the default resolves to None rather than raising.
    src = (
        "import os\nP = os.getenv('PORT', 8080)\nB = os.getenv('FLAG', True)\n"
    )
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert _kd(result) == {
        "PORT": [("getenv", None)],
        "FLAG": [("getenv", None)],
    }


def test_extract_env_vars_ignores_binary_files(tmp_path):
    # Finding 6: files that raise UnicodeDecodeError are skipped, not fatal.
    binary = tmp_path / "blob.py"
    binary.write_bytes(b'\xff\xfe\x00 not utf-8 os.getenv("X")')
    good = _write(tmp_path / "good.py", "import os\nx = os.getenv('GOOD')\n")

    result = m.extract_env_vars([binary, good])

    assert _kd(result) == {"GOOD": [("getenv", None)]}


def test_extract_env_vars_setdefault_captured_with_default(tmp_path):
    # v2: os.environ.setdefault("V", "d") is now scanned — the "d" would
    # otherwise be a hidden default a user editing .env.example has no way
    # to discover.
    src = (
        "import os\n"
        "os.environ.setdefault('GOOGLE_GENAI_USE_VERTEXAI', 'TRUE')\n"
        "os.environ.setdefault('NO_DEFAULT_HERE', 'x')\n"
    )
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert result["GOOGLE_GENAI_USE_VERTEXAI"][0].kind == "setdefault"
    assert result["GOOGLE_GENAI_USE_VERTEXAI"][0].default == "TRUE"
    assert result["NO_DEFAULT_HERE"][0].kind == "setdefault"
    assert result["NO_DEFAULT_HERE"][0].default == "x"


def test_extract_env_vars_records_line_number(tmp_path):
    # Provenance line numbers are captured so provenance comments can point
    # to a specific location if we ever surface them.
    src = "import os\n\n\nX = os.getenv('LINE_TEST', 'v')\n"
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    # The getenv call is on line 4.
    assert result["LINE_TEST"][0].line == 4


# ---------------------------------------------------------------------------
# AST detection edge cases — the scanner must find TRUE positives and skip
# TRUE negatives cleanly. False positives here would pollute .env.example;
# false negatives would silently fail to surface a var to the user.
# ---------------------------------------------------------------------------


def test_extract_env_vars_kwarg_default_captured(tmp_path):
    # `os.getenv("VAR", default="d")` — kwarg form. Rare but valid.
    # The scanner must capture "d" so the discoverability guarantee
    # holds regardless of calling style.
    src = (
        "import os\n"
        "a = os.getenv('VAR_A', default='kw-default')\n"
        "b = os.environ.get('VAR_B', default='kw-get-default')\n"
        "os.environ.setdefault('VAR_C', default='kw-sd-default')\n"
    )
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert result["VAR_A"][0].default == "kw-default"
    assert result["VAR_B"][0].default == "kw-get-default"
    assert result["VAR_C"][0].default == "kw-sd-default"


def test_extract_env_vars_positional_wins_over_kwarg_when_both_present(
    tmp_path,
):
    # Defensive: `os.getenv("X", "positional", default="kw")` is a
    # SyntaxError in Python (can't have both positional and kw for the
    # same param), but writers of weird code aside, our extractor should
    # prefer the positional when it comes to picking a source. Since
    # Python's own semantics reject this case at call time, we mainly
    # verify our code doesn't crash and picks something deterministic.
    #
    # Practically we test only the disambiguation shape our helper picks:
    # positional first, then kwarg. See _default_from_call.
    src = "import os\nx = os.getenv('X', 'pos')\n"
    py = _write(tmp_path / "mod.py", src)
    assert m.extract_env_vars([py])["X"][0].default == "pos"


def test_extract_env_vars_aliased_os_import_not_matched(tmp_path):
    # `import os as o` then `o.getenv(...)` — the AST walker only fires
    # on `os.getenv` (Attribute of Name('os')). This is intentional: we
    # can't reliably track aliases across a whole codebase, so we accept
    # a small false-negative rate on unusual import styles rather than
    # a false-positive rate on look-alike names.
    src = "import os as o\nx = o.getenv('SHOULD_NOT_MATCH', 'x')\n"
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert result == {}


def test_extract_env_vars_from_import_getenv_not_matched(tmp_path):
    # `from os import getenv` then `getenv(...)` — bare-name call. Same
    # reasoning as the aliased-import case: we require the `os.` prefix.
    src = "from os import getenv\nx = getenv('SHOULD_NOT_MATCH')\n"
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert result == {}


def test_extract_env_vars_similar_but_wrong_attribute_not_matched(tmp_path):
    # A non-os object with a `getenv` method must NOT match.
    src = (
        "class MyThing:\n"
        "    def getenv(self, key): return None\n"
        "obj = MyThing()\n"
        "x = obj.getenv('NOT_A_REAL_ENV_VAR')\n"
    )
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert result == {}


def test_extract_env_vars_nested_getenv_captures_inner(tmp_path):
    # `os.getenv(os.getenv("INNER", "def"), "outer-def")` — the inner
    # call is captured; the outer's var_name is None (its first arg is
    # not a string literal) so it doesn't contribute.
    src = "import os\nx = os.getenv(os.getenv('INNER', 'def'), 'outer')\n"
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert "INNER" in result
    assert result["INNER"][0].default == "def"
    # The outer call had a non-string first arg → no entry produced.
    # (There's nothing to assert absent by name — we just verify no
    # spurious UPPER_SNAKE_CASE var appeared.)
    assert len(result) == 1


def test_extract_env_vars_setdefault_without_default_still_captured(tmp_path):
    # `os.environ.setdefault("X")` is technically valid Python (returns
    # None if X isn't set; setting an env var to None then coerces to
    # "None" — nonsensical, but we still surface the var name).
    src = "import os\nos.environ.setdefault('LONELY_VAR')\n"
    py = _write(tmp_path / "mod.py", src)

    result = m.extract_env_vars([py])

    assert "LONELY_VAR" in result
    assert result["LONELY_VAR"][0].kind == "setdefault"
    assert result["LONELY_VAR"][0].default is None


def test_extract_env_vars_multiple_files_aggregated(tmp_path):
    # A var referenced in multiple files aggregates all sources into
    # one list (preserving per-file provenance for the resolver).
    f1 = _write(tmp_path / "a.py", "import os\nos.getenv('SHARED', 'from-a')\n")
    f2 = _write(tmp_path / "b.py", "import os\nos.getenv('SHARED', 'from-b')\n")

    result = m.extract_env_vars([f1, f2])

    assert len(result["SHARED"]) == 2
    files = {s.file for s in result["SHARED"]}
    assert files == {f1, f2}


def test_extract_env_vars_empty_file_list_is_empty_result():
    assert m.extract_env_vars([]) == {}


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


def _srcs(*items, file=None) -> list:
    """Build a list[Source] from concise (kind, default) tuples for tests.

    Every Source shares the same synthetic file (defaulting to a fake
    path — tests that care about file provenance override it).
    """
    fake = file if file is not None else Path("agent.py")
    return [
        m.Source(file=fake, kind=kind, default=default, line=1)
        for kind, default in items
    ]


def test_update_env_example_creates_file_when_absent(tmp_path):
    # v2: when a getenv/setdefault default is available in source, it IS
    # written to .env.example (with the marker comment). A var with no
    # source-side default still gets the TODO placeholder.
    env = tmp_path / ".env.example"

    added, upgraded = m.update_env_example(
        env,
        {
            "NEW_VAR": _srcs(("getenv", "val")),
            "NO_DEFAULT": _srcs(("getenv", None)),
        },
    )

    assert added == ["NEW_VAR", "NO_DEFAULT"]
    assert upgraded == []
    content = env.read_text(encoding="utf-8")
    # v2: real default extracted, with marker.
    assert "NEW_VAR=val" in content
    assert m.EXTRACTED_MARKER in content
    # No default in source → TODO placeholder with a "no default" note.
    assert f"NO_DEFAULT={m.PLACEHOLDER}" in content
    assert "no default in source" in content


def test_update_env_example_appends_only_missing(tmp_path):
    env = _write(tmp_path / ".env.example", "EXISTING=1\n")

    added, upgraded = m.update_env_example(
        env,
        {
            "EXISTING": _srcs(("getenv", None)),
            "FRESH": _srcs(("getenv", "2")),
        },
    )

    assert added == ["FRESH"]
    # EXISTING is USER_OWNED (no marker, value != v1 PLACEHOLDER) → not
    # touched, not counted as upgraded.
    assert upgraded == []
    content = env.read_text(encoding="utf-8")
    # Idempotency: the pre-existing EXISTING line is left exactly as it
    # was — NOT re-written, even though the skill now has a source for it.
    assert content.count("EXISTING") == 1
    assert "EXISTING=1\n" in content
    # The new FRESH line uses the extracted default (v2 behaviour).
    assert "FRESH=2" in content
    assert m.EXTRACTED_MARKER in content


def test_update_env_example_noop_when_all_present(tmp_path):
    env = _write(tmp_path / ".env.example", "A=1\nB=2\n")
    before = env.read_text(encoding="utf-8")

    added, upgraded = m.update_env_example(
        env,
        {"A": _srcs(("getenv", None)), "B": _srcs(("getenv", None))},
    )

    assert added == []
    assert upgraded == []
    assert env.read_text(encoding="utf-8") == before


def test_update_env_example_handles_missing_trailing_newline(tmp_path):
    # No trailing newline on the existing content must not merge lines.
    env = _write(tmp_path / ".env.example", "A=1")

    m.update_env_example(env, {"B": _srcs(("getenv", "2"))})

    lines = env.read_text(encoding="utf-8").splitlines()
    assert "A=1" in lines
    # v2: extracted default is written verbatim (with marker comment).
    assert any(ln.startswith("B=2") for ln in lines)


def test_update_env_example_dry_run_reports_but_does_not_write(tmp_path):
    env = tmp_path / ".env.example"

    added, upgraded = m.update_env_example(
        env, {"NEW": _srcs(("getenv", "1"))}, dry_run=True
    )

    # Reports what it would add ...
    assert added == ["NEW"]
    assert upgraded == []
    # ... but writes nothing (file not even created).
    assert not env.exists()


def test_update_env_example_writes_setdefault_value(tmp_path):
    # v2: os.environ.setdefault("V", "d") is the strongest form of author
    # commitment. The value MUST land in .env.example so users can find it.
    env = tmp_path / ".env.example"

    m.update_env_example(env, {"USE_VERTEX": _srcs(("setdefault", "TRUE"))})

    content = env.read_text(encoding="utf-8")
    assert "USE_VERTEX=TRUE" in content
    # Provenance comment names the kind.
    assert "setdefault" in content
    assert m.EXTRACTED_MARKER in content


def test_update_env_example_downgrades_placeholder_shaped_defaults(tmp_path):
    # v2: an obvious stub like "my-project-id" is a common pattern for
    # setdefault-as-reminder-to-fill-in. Writing it verbatim would give
    # users a value that runs but is wrong. Downgrade to TODO, but keep
    # the original string in the provenance comment so the maintainer
    # sees what was in source and can fix it.
    env = tmp_path / ".env.example"

    m.update_env_example(
        env,
        {"GOOGLE_CLOUD_PROJECT": _srcs(("setdefault", "my-project-id"))},
    )

    content = env.read_text(encoding="utf-8")
    # Downgraded to TODO ...
    assert f"GOOGLE_CLOUD_PROJECT={m.PLACEHOLDER}" in content
    # ... with the original value preserved in the marker comment.
    assert '"my-project-id"' in content
    # ... and no line that would resolve to the placeholder as-if-real.
    assert "GOOGLE_CLOUD_PROJECT=my-project-id" not in content


def test_update_env_example_preserves_user_edits_never_overwrites(tmp_path):
    # Golden rule: any USER_OWNED line — one without our marker AND a
    # value that isn't the v1 PLACEHOLDER — is left untouched. This is
    # what makes re-running the skill safe when a maintainer has typed
    # real values into .env.example.
    env = _write(
        tmp_path / ".env.example",
        "# hand-tuned by maintainer\nGOOGLE_CLOUD_PROJECT=my-real-project\n",
    )
    before = env.read_text(encoding="utf-8")

    added, upgraded = m.update_env_example(
        env,
        {"GOOGLE_CLOUD_PROJECT": _srcs(("setdefault", "something-else"))},
    )

    assert added == []
    assert upgraded == []
    # File not modified at all.
    assert env.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# _classify_entry / parse_env_example — 5-bucket classification
# ---------------------------------------------------------------------------


def _classify(raw_line: str):
    """Parse one line and return its classification (for concise assertions)."""
    entries = m._parse_env_content(raw_line)
    # Every test line here declares exactly one var.
    assert len(entries) == 1, f"expected one entry, got {list(entries)}"
    return next(iter(entries.values())).classification


def test_classify_bare_v1_todo_is_upgradable():
    # v1 skill wrote: `VAR=<TODO: update-this-value>` with no inline
    # comment. Recognisable as v1 by exact value match + no comment.
    line = f"GOOGLE_CLOUD_PROJECT={m.PLACEHOLDER}\n"
    assert _classify(line) == m.EntryClassification.V1_TODO


def test_classify_v1_todo_with_user_comment_is_user_owned():
    # Golden rule: even a TODO literal becomes USER_OWNED if the user
    # attached their own inline comment — that comment is real intent.
    line = (
        f"GOOGLE_CLOUD_PROJECT={m.PLACEHOLDER}"
        "  # my project is prj-foo, use gcloud config get project\n"
    )
    assert _classify(line) == m.EntryClassification.USER_OWNED


def test_classify_marker_with_placeholder_is_skill_todo():
    # A line the skill itself wrote with a TODO — marker present, value
    # is exactly PLACEHOLDER.
    line = (
        f"VAR={m.PLACEHOLDER}  # {m.EXTRACTED_MARKER}; no default in source\n"
    )
    assert _classify(line) == m.EntryClassification.SKILL_TODO


def test_classify_marker_with_real_value_is_skill_real():
    line = f"VAR=real-value  # {m.EXTRACTED_MARKER}; from setdefault in x.py\n"
    assert _classify(line) == m.EntryClassification.SKILL_REAL


def test_classify_user_typed_real_value_is_user_owned():
    line = "GOOGLE_API_KEY=YOUR_AI_STUDIO_API_KEY_HERE\n"
    assert _classify(line) == m.EntryClassification.USER_OWNED


def test_classify_user_typed_value_with_inline_comment_is_user_owned():
    line = "REGION=us-central1  # our default region\n"
    assert _classify(line) == m.EntryClassification.USER_OWNED


def test_parse_env_example_returns_line_index_for_rewrite(tmp_path):
    # The parser must record which physical line each entry is on, so
    # the rewriter can address it precisely.
    env = _write(
        tmp_path / ".env.example",
        f"# header\nA=1\n\nB={m.PLACEHOLDER}\nC=3\n",
    )
    entries = m.parse_env_example(env)
    # Zero-based line indices.
    assert entries["A"].line_index == 1
    assert entries["B"].line_index == 3
    assert entries["C"].line_index == 4


def test_parse_env_example_first_occurrence_wins(tmp_path):
    # A malformed .env.example with duplicate declarations of the same
    # var is common enough to defend against. First occurrence wins —
    # matches dotenv precedence AND means the rewriter has one
    # deterministic target.
    env = _write(
        tmp_path / ".env.example",
        f"DUP={m.PLACEHOLDER}\nDUP=second-value\n",
    )
    entries = m.parse_env_example(env)
    assert entries["DUP"].line_index == 0
    # First occurrence's classification carries — the second is ignored.
    assert entries["DUP"].classification == m.EntryClassification.V1_TODO


def test_classify_export_prefix_v1_todo():
    # `export VAR=<TODO: ...>` with no marker and no inline comment
    # should still classify as V1_TODO (export is orthogonal to the
    # classification signal).
    line = f"export FOO={m.PLACEHOLDER}\n"
    assert _classify(line) == m.EntryClassification.V1_TODO


def test_classify_export_prefix_marker_real():
    line = f"export FOO=real-value  # {m.EXTRACTED_MARKER}; from setdefault in x.py\n"
    assert _classify(line) == m.EntryClassification.SKILL_REAL


def test_classify_empty_value_is_user_owned():
    # `FOO=` (empty value) is user-owned — the user may have deliberately
    # set an empty default. It is NOT V1_TODO (that requires the exact
    # PLACEHOLDER string).
    line = "FOO=\n"
    assert _classify(line) == m.EntryClassification.USER_OWNED


def test_classify_value_with_whitespace_padding_still_v1_todo():
    # Whitespace around the value must be stripped before comparison —
    # otherwise `FOO=<TODO: update-this-value>   ` fails to classify
    # correctly and blocks the upgrade path.
    line = f"FOO=  {m.PLACEHOLDER}  \n"
    assert _classify(line) == m.EntryClassification.V1_TODO


def test_classify_indented_line_still_recognised():
    # Leading whitespace on a var line is valid per python-dotenv. The
    # classifier must not reject it — otherwise a well-formed but
    # indented .env.example would get spurious appends.
    line = f"  FOO={m.PLACEHOLDER}\n"
    assert _classify(line) == m.EntryClassification.V1_TODO


def test_parse_env_example_skips_blank_and_comment_only_lines(tmp_path):
    # Blank lines and comment-only lines aren't entries and must not
    # end up in the returned dict.
    env = _write(
        tmp_path / ".env.example",
        "\n# just a comment\n\nREAL=1\n   # indented comment\n\n",
    )
    entries = m.parse_env_example(env)
    assert set(entries.keys()) == {"REAL"}


def test_parse_env_example_skips_malformed_lines(tmp_path):
    # A line that doesn't match the env-var shape (no `=`, or wrong
    # name shape) is ignored — never crashes the parser.
    env = _write(
        tmp_path / ".env.example",
        "GOOD=1\n"
        "just a random line without equals\n"
        "lowercase_var=2\n"  # not UPPER_SNAKE_CASE
        "9NUMERIC_START=3\n"  # invalid identifier
        "ALSO_GOOD=4\n",
    )
    entries = m.parse_env_example(env)
    assert set(entries.keys()) == {"GOOD", "ALSO_GOOD"}


def test_parse_env_example_missing_file_returns_empty_dict(tmp_path):
    # No file → empty dict, not an exception. Downstream code relies
    # on this so it can be called unconditionally.
    entries = m.parse_env_example(tmp_path / "does-not-exist.env")
    assert entries == {}


def test_read_defined_vars_stays_backwards_compatible(tmp_path):
    # read_defined_vars is now a thin wrapper around parse_env_example.
    # Existing callers (assign_model_var_names) rely on the same return
    # shape — verify it still returns just names.
    env = _write(
        tmp_path / ".env.example",
        "A=1\nB=2\n# comment\n",
    )
    assert m.read_defined_vars(env) == {"A", "B"}


# ---------------------------------------------------------------------------
# _rewrite_var_line — safety-hardened in-place rewriter
# ---------------------------------------------------------------------------


def test_rewriter_replaces_the_matching_line_only():
    lines = ["A=1\n", "TARGET=old\n", "B=2\n"]
    ok = m._rewrite_var_line(lines, "TARGET", "TARGET=new  # marker")
    assert ok is True
    assert lines[0] == "A=1\n"
    assert lines[1] == "TARGET=new  # marker\n"
    assert lines[2] == "B=2\n"


def test_rewriter_preserves_export_prefix():
    # If the original had `export VAR=`, the rewrite keeps the prefix.
    lines = [f"export FOO={m.PLACEHOLDER}\n"]
    ok = m._rewrite_var_line(lines, "FOO", "FOO=real")
    assert ok is True
    assert lines[0] == "export FOO=real\n"


def test_rewriter_preserves_leading_indentation():
    # Golden rule: an indented line stays indented after rewrite. Some
    # .env.example files are hand-formatted with leading whitespace on
    # certain entries; the skill must not silently strip that.
    lines = [f"  FOO={m.PLACEHOLDER}\n"]
    ok = m._rewrite_var_line(lines, "FOO", "FOO=real")
    assert ok is True
    assert lines[0] == "  FOO=real\n"


def test_rewriter_preserves_indent_and_export_together():
    lines = [f"\texport FOO={m.PLACEHOLDER}\n"]
    ok = m._rewrite_var_line(lines, "FOO", "FOO=real")
    assert ok is True
    assert lines[0] == "\texport FOO=real\n"


def test_rewriter_empty_lines_list_returns_false():
    # No lines to match against → False, no crash.
    lines: list[str] = []
    assert m._rewrite_var_line(lines, "FOO", "FOO=x") is False


def test_rewriter_does_not_match_similar_prefixed_var_names():
    # `FOO` must not match `FOO_BAR` or `FOO_BAZ`. The regex uses
    # re.escape + anchored end, so this is regression-guarded.
    lines = [
        "FOO_BAR=1\n",
        "FOO_BAZ=2\n",
        "FOO=target\n",
    ]
    ok = m._rewrite_var_line(lines, "FOO", "FOO=new")
    assert ok is True
    # Only the FOO line changed.
    assert lines[0] == "FOO_BAR=1\n"
    assert lines[1] == "FOO_BAZ=2\n"
    assert lines[2] == "FOO=new\n"


def test_rewriter_leaves_all_non_matching_lines_byte_identical():
    # Regression guard: replacing one line must never mutate any other
    # line — including whitespace, comments, encoding, or ordering.
    lines = [
        "# License block\n",
        "# ------------\n",
        "\n",
        "ALPHA=1\n",
        "\r\n",  # CRLF blank
        "BETA=2  # a user note\n",
        f"TARGET={m.PLACEHOLDER}\n",
        "GAMMA=3\n",
        "\n",
    ]
    snapshot = list(lines)
    ok = m._rewrite_var_line(lines, "TARGET", "TARGET=real")
    assert ok is True
    for i, original in enumerate(snapshot):
        if i == 6:  # the TARGET line
            assert lines[i] != original
            assert lines[i] == "TARGET=real\n"
        else:
            assert lines[i] == original, f"line {i} unexpectedly changed"


def test_rewriter_handles_tab_around_equals():
    # `FOO\t=\tvalue` — legal per python-dotenv (which permits
    # whitespace around `=`). Our regex allows [ \t]* on both sides.
    lines = [f"FOO\t=\t{m.PLACEHOLDER}\n"]
    ok = m._rewrite_var_line(lines, "FOO", "FOO=real")
    assert ok is True
    # We use the standard shape on output (single `=`, no tabs).
    # Preserved: nothing about the whitespace within the line.
    assert lines[0] == "FOO=real\n"


def test_rewriter_no_newline_on_last_line():
    # Last line of file might have no trailing newline. The rewriter
    # must preserve that — adding a newline could subtly change tools
    # that care (git diff noise, some parsers).
    lines = [f"FOO={m.PLACEHOLDER}"]  # no trailing newline
    ok = m._rewrite_var_line(lines, "FOO", "FOO=real")
    assert ok is True
    assert lines[0] == "FOO=real"


def test_rewriter_refuses_quoted_value():
    # Quoted values could span lines / contain escapes / include the
    # `#` character verbatim — anything the classifier's simple regex
    # doesn't handle unambiguously. Fail closed.
    lines = ['FOO="quoted value"\n']
    ok = m._rewrite_var_line(lines, "FOO", "FOO=new")
    assert ok is False
    # File-analog left untouched.
    assert lines[0] == 'FOO="quoted value"\n'


def test_rewriter_refuses_line_continuation():
    # Backslash-continuation is not standard dotenv but some parsers
    # accept it. If we see it, we can't safely rewrite one physical line.
    lines = ["FOO=part_one\\\n", "part_two\n"]
    ok = m._rewrite_var_line(lines, "FOO", "FOO=new")
    assert ok is False


def test_rewriter_refuses_multiple_matches():
    # If the file has two lines declaring FOO, we don't know which one
    # is the "real" declaration. Refuse rather than guess.
    lines = ["FOO=1\n", "FOO=2\n"]
    ok = m._rewrite_var_line(lines, "FOO", "FOO=3")
    assert ok is False
    # Both original lines untouched.
    assert lines == ["FOO=1\n", "FOO=2\n"]


def test_rewriter_refuses_when_var_absent():
    lines = ["A=1\n", "B=2\n"]
    ok = m._rewrite_var_line(lines, "NOT_HERE", "NOT_HERE=x")
    assert ok is False


def test_rewriter_preserves_crlf_newlines():
    # Windows line endings survive rewriting verbatim.
    lines = ["FOO=old\r\n"]
    ok = m._rewrite_var_line(lines, "FOO", "FOO=new")
    assert ok is True
    assert lines[0] == "FOO=new\r\n"


# ---------------------------------------------------------------------------
# update_env_example — v2.1 upgrade paths (in-place TODO -> real value)
# ---------------------------------------------------------------------------


def test_upgrade_v1_todo_to_real_setdefault_value(tmp_path):
    # The headline case: recipe was processed by v1 (line has PLACEHOLDER,
    # no marker), source now has a setdefault with a real value. The v1
    # TODO should be rewritten in place to the real value.
    env = _write(
        tmp_path / ".env.example",
        "# Environment variables extracted by extract-python-environment-variables\n"
        f"USE_VERTEX={m.PLACEHOLDER}\n",
    )

    added, upgraded = m.update_env_example(
        env, {"USE_VERTEX": _srcs(("setdefault", "TRUE"))}
    )

    assert added == []
    assert upgraded == ["USE_VERTEX"]
    content = env.read_text(encoding="utf-8")
    assert "USE_VERTEX=TRUE" in content
    assert m.EXTRACTED_MARKER in content
    # v1 header comment is preserved (rewriter touches one line, not the
    # surrounding block).
    assert (
        "Environment variables extracted by extract-python-environment-variables"
        in content
    )
    # No stray TODO for this var left behind.
    assert f"USE_VERTEX={m.PLACEHOLDER}" not in content


def test_upgrade_skill_todo_to_real_value(tmp_path):
    # v2 skill previously wrote a TODO (because source had no default at
    # the time). Now source has a real default — upgrade.
    env = _write(
        tmp_path / ".env.example",
        f"USE_VERTEX={m.PLACEHOLDER}  # {m.EXTRACTED_MARKER}; no default in source\n",
    )

    added, upgraded = m.update_env_example(
        env, {"USE_VERTEX": _srcs(("setdefault", "TRUE"))}
    )

    assert added == []
    assert upgraded == ["USE_VERTEX"]
    content = env.read_text(encoding="utf-8")
    assert "USE_VERTEX=TRUE" in content


def test_do_not_upgrade_when_source_default_is_placeholder_shape(tmp_path):
    # Correctness: source has "my-project-id" (placeholder-shaped). Even
    # though the existing entry is a v1 TODO, upgrading it would produce
    # another TODO — pure churn. Skip.
    env = _write(
        tmp_path / ".env.example",
        f"GOOGLE_CLOUD_PROJECT={m.PLACEHOLDER}\n",
    )
    before = env.read_text(encoding="utf-8")

    added, upgraded = m.update_env_example(
        env,
        {"GOOGLE_CLOUD_PROJECT": _srcs(("setdefault", "my-project-id"))},
    )

    assert added == []
    assert upgraded == []
    # File byte-for-byte unchanged.
    assert env.read_text(encoding="utf-8") == before


def test_do_not_upgrade_when_source_has_no_default(tmp_path):
    # Source only has bare os.getenv("V") — no string default to upgrade
    # TO. Leave the TODO alone (upgrading to another TODO is churn).
    env = _write(
        tmp_path / ".env.example",
        f"NO_DEFAULT={m.PLACEHOLDER}\n",
    )
    before = env.read_text(encoding="utf-8")

    added, upgraded = m.update_env_example(
        env, {"NO_DEFAULT": _srcs(("getenv", None))}
    )

    assert added == []
    assert upgraded == []
    assert env.read_text(encoding="utf-8") == before


def test_do_not_upgrade_v1_todo_with_inline_comment(tmp_path):
    # Golden rule: an inline comment on a TODO line means the user has
    # engaged with it. Never touch, even if source has a real default.
    env = _write(
        tmp_path / ".env.example",
        f"GOOGLE_CLOUD_PROJECT={m.PLACEHOLDER}  # my note about this var\n",
    )
    before = env.read_text(encoding="utf-8")

    added, upgraded = m.update_env_example(
        env,
        {"GOOGLE_CLOUD_PROJECT": _srcs(("setdefault", "real-project"))},
    )

    assert added == []
    assert upgraded == []
    assert env.read_text(encoding="utf-8") == before


def test_do_not_upgrade_skill_real_on_source_drift(tmp_path):
    # Skill previously wrote a real value. Source now says something
    # different. Skip refresh — the maintainer may have relied on the
    # persisted value; drift is a separate concern (surface via WARN
    # elsewhere).
    env = _write(
        tmp_path / ".env.example",
        f"LOG_LEVEL=INFO  # {m.EXTRACTED_MARKER}; from getenv in x.py\n",
    )
    before = env.read_text(encoding="utf-8")

    added, upgraded = m.update_env_example(
        env, {"LOG_LEVEL": _srcs(("getenv", "DEBUG"))}
    )

    assert added == []
    assert upgraded == []
    assert env.read_text(encoding="utf-8") == before


def test_upgrade_and_append_can_happen_in_one_pass(tmp_path):
    # Realistic mixed case: some vars exist as v1 TODOs (upgradable),
    # some are missing (appendable), some are user-owned (skip).
    env = _write(
        tmp_path / ".env.example",
        f"UPGRADABLE={m.PLACEHOLDER}\nUSER_OWNED_VAR=my-real-value\n",
    )

    added, upgraded = m.update_env_example(
        env,
        {
            "UPGRADABLE": _srcs(("setdefault", "TRUE")),
            "USER_OWNED_VAR": _srcs(("setdefault", "should-not-win")),
            "APPEND_ME": _srcs(("getenv", "hello")),
        },
    )

    assert added == ["APPEND_ME"]
    assert upgraded == ["UPGRADABLE"]
    content = env.read_text(encoding="utf-8")
    assert "UPGRADABLE=TRUE" in content
    assert "APPEND_ME=hello" in content
    # User-owned entry untouched.
    assert "USER_OWNED_VAR=my-real-value" in content
    assert "should-not-win" not in content


def test_upgrade_is_idempotent(tmp_path):
    # Golden rule: running the skill twice in a row must produce zero
    # additional changes on the second run.
    env = _write(
        tmp_path / ".env.example",
        f"V={m.PLACEHOLDER}\n",
    )

    _added1, upgraded1 = m.update_env_example(
        env, {"V": _srcs(("setdefault", "real"))}
    )
    content_after_first = env.read_text(encoding="utf-8")

    added2, upgraded2 = m.update_env_example(
        env, {"V": _srcs(("setdefault", "real"))}
    )
    content_after_second = env.read_text(encoding="utf-8")

    assert upgraded1 == ["V"]
    assert upgraded2 == []  # already upgraded → SKILL_REAL → skip
    assert added2 == []
    assert content_after_first == content_after_second


def test_upgrade_dry_run_reports_but_does_not_write(tmp_path):
    env = _write(
        tmp_path / ".env.example",
        f"V={m.PLACEHOLDER}\n",
    )
    before = env.read_text(encoding="utf-8")

    _added, upgraded = m.update_env_example(
        env, {"V": _srcs(("setdefault", "TRUE"))}, dry_run=True
    )

    assert upgraded == ["V"]
    # File left untouched.
    assert env.read_text(encoding="utf-8") == before


def test_upgrade_refuses_ambiguous_line_and_warns(tmp_path, capsys):
    # Golden rule: an existing entry classified as V1_TODO but whose
    # line is structurally ambiguous (here: two matching lines for the
    # same var) must not be rewritten. Instead, warn and leave it alone.
    env = _write(
        tmp_path / ".env.example",
        f"V={m.PLACEHOLDER}\nV={m.PLACEHOLDER}\n",
    )
    before = env.read_text(encoding="utf-8")

    # Parser gives us the first V (V1_TODO), so we plan to upgrade.
    # Rewriter finds two matches, refuses, and warns.
    _added, upgraded = m.update_env_example(
        env, {"V": _srcs(("setdefault", "TRUE"))}
    )

    assert upgraded == []
    err = capsys.readouterr().err
    assert "Refused to upgrade V" in err
    # File not clobbered.
    assert env.read_text(encoding="utf-8") == before


def test_upgrade_preserves_ordering_and_surrounding_lines(tmp_path):
    # Regression guard: rewriting one line must not touch anything else
    # in the file — blank lines, comment blocks, order.
    env = _write(
        tmp_path / ".env.example",
        "# Section 1\n"
        "# User's careful comment above the TODO\n"
        f"UPGRADABLE={m.PLACEHOLDER}\n"
        "\n"
        "# Section 2\n"
        "OTHER=leave-me-alone\n",
    )

    m.update_env_example(env, {"UPGRADABLE": _srcs(("setdefault", "TRUE"))})

    content = env.read_text(encoding="utf-8")
    lines = content.splitlines()
    # The user's comment above the TODO is preserved verbatim.
    assert "# User's careful comment above the TODO" in lines
    # Order of surrounding sections preserved.
    upgradable_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("UPGRADABLE=")
    )
    other_idx = next(i for i, ln in enumerate(lines) if ln.startswith("OTHER="))
    assert upgradable_idx < other_idx
    # Other var untouched.
    assert "OTHER=leave-me-alone" in lines


def test_update_env_example_noop_when_env_vars_empty(tmp_path):
    # A recipe with zero env-var accesses → nothing to do, no file
    # touched, no crash.
    env = _write(tmp_path / ".env.example", "EXISTING=1\n")
    before = env.read_text(encoding="utf-8")
    added, upgraded = m.update_env_example(env, {})
    assert (added, upgraded) == ([], [])
    assert env.read_text(encoding="utf-8") == before


def test_update_env_example_noop_when_no_changes_needed_does_not_touch_file(
    tmp_path,
):
    # Every var already declared + no upgrades required → the file must
    # not be re-written at all (would show as a spurious modification in
    # git status / re-run diff).
    env = _write(tmp_path / ".env.example", "A=user-value\n")
    original_mtime = env.stat().st_mtime_ns
    original_bytes = env.read_bytes()

    added, upgraded = m.update_env_example(
        env, {"A": _srcs(("getenv", "different"))}
    )
    assert (added, upgraded) == ([], [])
    # File content byte-identical.
    assert env.read_bytes() == original_bytes
    # (mtime check is best-effort — some filesystems have coarse
    # resolution — but the byte-equality assertion is the strong one.)
    _ = original_mtime  # silence unused


def test_update_env_example_round_trip_check_refuses_corrupted_rewrite(
    tmp_path, monkeypatch, capsys
):
    # SAFETY BACKSTOP: if a hypothetical bug in the rewriter produced
    # a line that doesn't re-parse as SKILL_REAL, the round-trip check
    # must catch it and refuse to write. This test forces that by
    # monkeypatching the rewriter to succeed structurally but produce
    # a corrupt line, and asserts the whole write is rolled back.
    env = _write(tmp_path / ".env.example", f"V={m.PLACEHOLDER}\n")
    before = env.read_bytes()

    def _corrupt_rewriter(lines, var_name, new_line_body):
        # Simulate a rewriter bug: signal success but write garbage
        # that will never re-parse as a valid entry.
        lines[0] = "totally not a valid env line !!!\n"
        return True

    monkeypatch.setattr(m, "_rewrite_var_line", _corrupt_rewriter)

    added, upgraded = m.update_env_example(
        env, {"V": _srcs(("setdefault", "real"))}
    )
    # Writer refused because of the round-trip check.
    assert (added, upgraded) == ([], [])
    # File byte-identical.
    assert env.read_bytes() == before
    # ERROR was logged so a caller / CI can see it.
    err = capsys.readouterr().err
    assert "Round-trip safety check failed" in err


def test_update_env_example_upgrade_only_no_append_still_writes(tmp_path):
    # A recipe where every var is already declared but some need
    # upgrading — the writer must still update the file (upgrade
    # path is not gated on there being appends).
    env = _write(tmp_path / ".env.example", f"V={m.PLACEHOLDER}\n")
    added, upgraded = m.update_env_example(
        env, {"V": _srcs(("setdefault", "real"))}
    )
    assert added == []
    assert upgraded == ["V"]
    assert "V=real" in env.read_text(encoding="utf-8")


def test_update_env_example_preserves_crlf_line_endings(tmp_path):
    # A .env.example authored on Windows should keep its CRLF endings
    # after an in-place upgrade — mixed endings in one file are ugly
    # and can break naive tools.
    env = tmp_path / ".env.example"
    env.write_bytes(f"V={m.PLACEHOLDER}\r\n".encode())

    m.update_env_example(env, {"V": _srcs(("setdefault", "real"))})

    # Line still ends with CRLF, not LF.
    content_bytes = env.read_bytes()
    assert b"\r\n" in content_bytes
    assert b"V=real" in content_bytes


# ---------------------------------------------------------------------------
# looks_like_placeholder
# ---------------------------------------------------------------------------


def test_looks_like_placeholder_detects_empty_string():
    assert m.looks_like_placeholder("") is not None


def test_looks_like_placeholder_detects_angle_wrapped():
    assert m.looks_like_placeholder("<TODO>") is not None
    assert m.looks_like_placeholder("<your-key-here>") is not None


def test_looks_like_placeholder_detects_generic_tokens():
    # Exact-match tokens are always placeholders.
    for token in ("changeme", "change-me", "TODO", "todo", "xxx", "foo"):
        assert m.looks_like_placeholder(token) is not None, token


def test_looks_like_placeholder_detects_your_and_my_prefixes():
    assert m.looks_like_placeholder("your-api-key") is not None
    assert m.looks_like_placeholder("YOUR_PROJECT") is not None
    assert m.looks_like_placeholder("my-project-id") is not None
    assert m.looks_like_placeholder("MY_KEY") is not None


def test_looks_like_placeholder_detects_example_com():
    assert m.looks_like_placeholder("https://api.example.com") is not None


def test_looks_like_placeholder_passes_real_looking_values():
    # Values that look like plausible real config MUST NOT be flagged, or
    # the writer downgrades a real default to TODO — worse than doing
    # nothing.
    for value in (
        "TRUE",
        "INFO",
        "us-central1",
        "gemini-3.5-flash",  # a real model id
        "1234567890",
        "https://api.acme.corp/v1",
        "postgresql://localhost:5432/db",
    ):
        assert m.looks_like_placeholder(value) is None, value


def test_looks_like_placeholder_broader_realistic_defaults():
    # Extended coverage: things that should PASS through. If any of these
    # get downgraded to TODO, real config will be missed.
    for value in (
        "0",
        "1",
        "false",
        "False",
        "off",
        "DEBUG",
        "ERROR",
        "utf-8",
        "en_US.UTF-8",
        "8080",
        "/tmp/cache",
        "s3://bucket/prefix/",
        "redis://cache.local:6379/0",
        "us-east4-a",
        "text-embedding-004",
        "gs://my-bucket/path",
        "@channel-name",
        "prod",
        "staging",
    ):
        assert m.looks_like_placeholder(value) is None, value


def test_looks_like_placeholder_your_and_my_only_with_delimiter():
    # We only flag "your-" / "your_" / "my-" / "my_" — not any word
    # starting with those substrings (e.g. "yours", "yourself", "myth",
    # "myself"). This prevents false positives on legitimate words.
    for value in ("yours", "yourself", "myth", "myself", "mysql", "yourfoo"):
        assert m.looks_like_placeholder(value) is None, value


def test_looks_like_placeholder_case_insensitive_prefix_matching():
    # "your-" prefix check is case-insensitive so YOUR_KEY / your-key /
    # YoUr-KeY all get flagged.
    for value in ("YOUR-KEY", "yOUr-key", "MY-project", "Your_secret"):
        assert m.looks_like_placeholder(value) is not None, value


def test_looks_like_placeholder_example_com_in_url_flagged():
    # example.com anywhere in the value triggers the flag — this is
    # deliberate. A URL with example.com is documentation-shaped, not
    # a real endpoint.
    assert m.looks_like_placeholder("https://api.example.com/v1") is not None
    assert m.looks_like_placeholder("user@example.com") is not None


def test_looks_like_placeholder_returns_reason_string():
    # The return value must be a non-empty reason string, not just a
    # truthy sentinel — the writer surfaces it in the marker comment.
    reason = m.looks_like_placeholder("")
    assert isinstance(reason, str) and reason
    reason = m.looks_like_placeholder("changeme")
    assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# resolve_default
# ---------------------------------------------------------------------------


def test_resolve_default_returns_none_when_no_string_default():
    # A var read only via os.environ["V"] / bare os.getenv("V") has no
    # string default anywhere — resolve should return None to signal
    # "write TODO".
    sources = _srcs(("environ_sub", None), ("getenv", None))
    resolved = m.resolve_default(sources)
    assert resolved.value is None
    assert resolved.winner is None
    assert resolved.conflict_count == 0


def test_resolve_default_setdefault_beats_getenv_fallback():
    # Priority: setdefault (author committed to a process-wide default)
    # wins over per-read getenv fallback.
    sources = [
        m.Source(Path("a.py"), "getenv", "from-getenv", 1),
        m.Source(Path("b.py"), "setdefault", "from-setdefault", 1),
    ]
    resolved = m.resolve_default(sources)
    assert resolved.value == "from-setdefault"
    assert resolved.winner is not None
    assert resolved.winner.kind == "setdefault"
    # They disagreed, so the getenv default counts as one conflict.
    assert resolved.conflict_count == 1


def test_resolve_default_alphabetical_file_tiebreak_within_tier():
    # Two setdefault sources with different values: earlier-file wins.
    sources = [
        m.Source(Path("zzz.py"), "setdefault", "z-value", 1),
        m.Source(Path("aaa.py"), "setdefault", "a-value", 1),
    ]
    resolved = m.resolve_default(sources)
    assert resolved.value == "a-value"
    assert resolved.winner is not None
    assert resolved.winner.file == Path("aaa.py")


def test_resolve_default_conflict_count_ignores_agreeing_sources():
    # Two sources with the same default are not a conflict.
    sources = [
        m.Source(Path("a.py"), "getenv", "shared", 1),
        m.Source(Path("b.py"), "getenv", "shared", 1),
    ]
    resolved = m.resolve_default(sources)
    assert resolved.value == "shared"
    assert resolved.conflict_count == 0


def test_resolve_default_ignores_sources_without_string_default():
    # A bare os.getenv("V") alongside an os.getenv("V", "x") should not
    # be counted as a conflict — it doesn't propose a value.
    sources = [
        m.Source(Path("a.py"), "getenv", None, 1),
        m.Source(Path("b.py"), "getenv", "the-value", 1),
    ]
    resolved = m.resolve_default(sources)
    assert resolved.value == "the-value"
    assert resolved.conflict_count == 0


def test_resolve_default_empty_sources_returns_none():
    # Defensive: an empty source list must not crash.
    resolved = m.resolve_default([])
    assert resolved.value is None
    assert resolved.winner is None
    assert resolved.conflict_count == 0


def test_resolve_default_line_number_tiebreak_within_same_file():
    # Two sources in the SAME file with different defaults: the earlier
    # line wins (deterministic ordering after (kind, file, line)).
    sources = [
        m.Source(Path("x.py"), "setdefault", "later", 50),
        m.Source(Path("x.py"), "setdefault", "earlier", 5),
    ]
    resolved = m.resolve_default(sources)
    assert resolved.value == "earlier"
    assert resolved.winner is not None
    assert resolved.winner.line == 5


def test_resolve_default_environ_get_and_getenv_at_same_priority():
    # environ_get and getenv share the SAME priority tier (1); a
    # setdefault (tier 0) beats both. Within tier 1, alphabetical
    # file wins.
    sources = [
        m.Source(Path("b.py"), "getenv", "b-value", 1),
        m.Source(Path("a.py"), "environ_get", "a-value", 1),
    ]
    resolved = m.resolve_default(sources)
    # Alphabetical: "a.py" < "b.py" → a-value wins.
    assert resolved.value == "a-value"


def test_resolve_default_environ_sub_never_contributes_value():
    # environ_sub sources always have default=None, so they can never
    # supply a winning value. If ALL sources are environ_sub, the
    # resolver returns None.
    sources = [
        m.Source(Path("a.py"), "environ_sub", None, 1),
        m.Source(Path("b.py"), "environ_sub", None, 1),
    ]
    resolved = m.resolve_default(sources)
    assert resolved.value is None


def test_resolve_default_setdefault_wins_over_environ_get():
    # setdefault is tier 0; environ_get is tier 1. setdefault always wins.
    sources = [
        m.Source(Path("z.py"), "setdefault", "sd-value", 1),
        m.Source(
            Path("a.py"), "environ_get", "get-value", 1
        ),  # earlier alphabetically!
    ]
    resolved = m.resolve_default(sources)
    # Priority beats alphabetical.
    assert resolved.value == "sd-value"


# ---------------------------------------------------------------------------
# format_env_entry
# ---------------------------------------------------------------------------


def test_format_env_entry_writes_todo_when_no_default():
    # No source supplied a string-literal default.
    line = m.format_env_entry("API_KEY", _srcs(("getenv", None)))
    assert line.startswith(f"API_KEY={m.PLACEHOLDER}")
    assert m.EXTRACTED_MARKER in line
    assert "no default in source" in line


def test_format_env_entry_writes_real_value_with_provenance(tmp_path):
    # Extracted default becomes the value; provenance names the kind and
    # (relative) file path.
    src_file = tmp_path / "sub" / "agent.py"
    sources = [m.Source(src_file, "setdefault", "TRUE", 3)]
    line = m.format_env_entry("USE_X", sources, recipe_dir=tmp_path)
    assert line.startswith("USE_X=TRUE")
    assert "setdefault" in line
    assert "sub/agent.py" in line
    assert m.EXTRACTED_MARKER in line


def test_format_env_entry_downgrades_placeholder_shaped_value(tmp_path):
    # A stub-shaped value → TODO with the source string preserved in the
    # comment so the maintainer can find and fix the source too.
    src_file = tmp_path / "agent.py"
    sources = [m.Source(src_file, "setdefault", "my-project-id", 1)]
    line = m.format_env_entry("PROJECT", sources, recipe_dir=tmp_path)
    assert line.startswith(f"PROJECT={m.PLACEHOLDER}")
    assert '"my-project-id"' in line
    # And there's a hint that says please fix source.
    assert "fix source" in line


def test_format_env_entry_records_conflict_in_comment(tmp_path):
    # Two conflicting sources → note the conflict so maintainer reconciles.
    sources = [
        m.Source(tmp_path / "a.py", "setdefault", "TRUE", 1),
        m.Source(tmp_path / "b.py", "setdefault", "FALSE", 1),
    ]
    line = m.format_env_entry("FLAG", sources, recipe_dir=tmp_path)
    # Alphabetical tie-break within setdefault tier picks a.py's TRUE.
    assert line.startswith("FLAG=TRUE")
    assert "differs from 1 other source" in line


def test_format_env_entry_no_conflict_note_when_sources_agree(tmp_path):
    # Two sources with the same default → no conflict note in the comment
    # (the comment should stay clean when nothing needs reconciling).
    sources = [
        m.Source(tmp_path / "a.py", "setdefault", "TRUE", 1),
        m.Source(tmp_path / "b.py", "setdefault", "TRUE", 1),
    ]
    line = m.format_env_entry("FLAG", sources, recipe_dir=tmp_path)
    assert line.startswith("FLAG=TRUE")
    assert "differs from" not in line


def test_format_env_entry_uses_basename_when_no_recipe_dir(tmp_path):
    # Callers can pass recipe_dir=None (the parameter is optional). When
    # they do, provenance uses the file's basename, not an absolute path.
    src_file = tmp_path / "deep" / "nested" / "agent.py"
    sources = [m.Source(src_file, "setdefault", "value", 1)]
    line = m.format_env_entry("V", sources)  # recipe_dir omitted
    assert "agent.py" in line
    assert str(tmp_path) not in line  # no absolute path leaked


def test_format_env_entry_uses_basename_when_file_not_under_recipe_dir(
    tmp_path,
):
    # If the source file is somewhere else on disk (unusual), we fall
    # back to basename rather than emitting an absolute path.
    src_file = Path("/tmp/somewhere/else.py")
    sources = [m.Source(src_file, "setdefault", "value", 1)]
    line = m.format_env_entry("V", sources, recipe_dir=tmp_path)
    assert "else.py" in line
    assert "/tmp/somewhere" not in line


def test_format_env_entry_line_contains_no_newlines():
    # The formatter returns a SINGLE line; callers add the newline. A
    # returned string with embedded newlines would corrupt .env.example.
    sources = [m.Source(Path("x.py"), "setdefault", "value", 1)]
    line = m.format_env_entry("V", sources)
    assert "\n" not in line
    assert "\r" not in line


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
    # still add a "noqa E402" marker to the trailing relative import — otherwise
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


def test_assign_model_var_names_single():
    assert m.assign_model_var_names({"gemini-3.5-flash"}) == {
        "gemini-3.5-flash": "MODEL_NAME"
    }


def test_assign_model_var_names_multiple():
    # Multiple models get MODEL_NAME_GENERATED_N (sorted for determinism).
    # Sorted order: "claude-3-sonnet" < "gemini-3.5-flash".
    result = m.assign_model_var_names({"gemini-3.5-flash", "claude-3-sonnet"})
    assert result == {
        "claude-3-sonnet": "MODEL_NAME_GENERATED_1",
        "gemini-3.5-flash": "MODEL_NAME_GENERATED_2",
    }


def test_assign_model_var_names_skips_taken_counters():
    # When MODEL_NAME is already taken, the counter scheme is used.
    # If MODEL_NAME_GENERATED_1 is also taken, the first available slot is _2.
    result = m.assign_model_var_names(
        {"gemini-3.5-flash"},
        existing_vars={"MODEL_NAME", "MODEL_NAME_GENERATED_1"},
    )
    assert result == {"gemini-3.5-flash": "MODEL_NAME_GENERATED_2"}


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


def test_extract_hardcoded_models_skips_getenv_positional_default(tmp_path):
    # REGRESSION: a string literal serving as the DEFAULT argument of an
    # existing os.getenv() call must NOT be treated as a "raw hardcoded
    # literal that needs promotion." The maintainer already made the var
    # env-configurable and picked this string as the fallback; replacing
    # it with bare os.getenv("MODEL_NAME") (which returns None) would
    # silently regress the type from str to str | None and break at
    # runtime when both env vars are unset.
    src = (
        "import os\n"
        "MODEL = os.getenv('MODEL_NAME_GENERATED_1', 'gemini-3.5-flash')\n"
    )
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])

    # Nothing to replace — the model literal was already serving as a
    # getenv default, so it's already env-var-controlled.
    assert py not in hits


def test_extract_hardcoded_models_skips_getenv_kwarg_default(tmp_path):
    # Same protection for the keyword form: os.getenv("X", default="y").
    src = (
        "import os\n"
        "M = os.getenv('MODEL_NAME_GENERATED_1', default='gemini-3.5-flash')\n"
    )
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    assert py not in hits


def test_extract_hardcoded_models_skips_environ_get_default(tmp_path):
    # os.environ.get("X", "d") — same protection.
    src = "import os\nM = os.environ.get('MODEL_NAME', 'gemini-3.5-flash')\n"
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    assert py not in hits


def test_extract_hardcoded_models_skips_environ_setdefault_default(tmp_path):
    # os.environ.setdefault("X", "d") — same protection.
    src = "import os\nos.environ.setdefault('MODEL_NAME', 'gemini-3.5-flash')\n"
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    assert py not in hits


def test_extract_hardcoded_models_still_detects_true_hardcoded_kwarg(tmp_path):
    # Negative: the SECOND arg of a NON-getenv call is fair game. This is
    # exactly the case the model-replacement path was designed for:
    #   Agent(name="x", model="gemini-3.5-flash")
    # The string is truly hardcoded here and SHOULD be extracted.
    src = 'agent = Agent(name="x", model="gemini-3.5-flash")\n'
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    assert py in hits
    assert [s for _l, s in hits[py]] == ["gemini-3.5-flash"]


def test_extract_hardcoded_models_still_detects_direct_assignment(tmp_path):
    # Negative: a plain assignment like `MODEL = "gemini-3.5-flash"` is
    # still a hardcoded literal (not inside any getenv call). Detected.
    src = 'MODEL = "gemini-3.5-flash"\n'
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    assert py in hits
    assert [s for _l, s in hits[py]] == ["gemini-3.5-flash"]


def test_extract_hardcoded_models_mixed_hardcoded_and_getenv_defaults(tmp_path):
    # A file with BOTH a true hardcoded literal AND a getenv-default literal.
    # Only the true hardcoded one is picked up.
    src = (
        "import os\n"
        'HARDCODED = "gemini-3.5-flash"\n'
        'FROM_ENV = os.getenv("VAR", "gemini-3.5-flash")\n'
    )
    py = _write(tmp_path / "mod.py", src)

    hits = m.extract_hardcoded_models([py])
    # Only ONE hit — the direct assignment. The getenv default is skipped.
    assert py in hits
    assert len(hits[py]) == 1
    _lineno, model_str = hits[py][0]
    assert model_str == "gemini-3.5-flash"


def test_replace_hardcoded_models_does_not_touch_getenv_defaults(tmp_path):
    # End-to-end regression: the write path must respect the exclusion too,
    # even if a caller somehow constructed a `hits` dict that included a
    # getenv-default node. Concretely: run the whole pipeline against a
    # file that has ONLY getenv-default model literals, and verify the file
    # is byte-identical after the run.
    src = (
        "import os\n"
        "M1 = os.getenv('MODEL_NAME_GENERATED_1', 'gemini-3.5-flash')\n"
        'M2 = os.environ.get("MODEL_NAME_GENERATED_2", "gemini-3.5-flash")\n'
    )
    py = _write(tmp_path / "mod.py", src)
    before = py.read_bytes()

    hits = m.extract_hardcoded_models([py])
    # extract_hardcoded_models returns empty → nothing to substitute →
    # no name_map entries → no write attempt.
    assert hits == {}
    # But if a caller (or a bug) DID try to replace, the replacement path
    # itself would still skip via the same exclusion. Simulate that here:
    forced_hits = {py: [(2, "gemini-3.5-flash"), (3, "gemini-3.5-flash")]}
    name_map = m.assign_model_var_names({"gemini-3.5-flash"})
    substituted = m.replace_hardcoded_models(
        py_files=[py], hits=forced_hits, name_map=name_map
    )
    # No substitution happened because both hits were getenv defaults.
    assert substituted == {}
    # File byte-identical.
    assert py.read_bytes() == before


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
    # Sorted: "claude-3-sonnet" → GENERATED_1, "gemini-3.5-flash" → GENERATED_2.
    # Both duplicate occurrences of gemini-3.5-flash map to the same env var.
    assert content.count('os.getenv("MODEL_NAME_GENERATED_2")') == 2
    assert content.count('os.getenv("MODEL_NAME_GENERATED_1")') == 1
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
    """A minimal recipe exercising all v2 code paths:

    * MY_API_KEY — bare os.getenv, no default → TODO
    * LOG_LEVEL — getenv with a real fallback → extracted verbatim
    * USE_VERTEX — setdefault with a real value → extracted verbatim
    * GOOGLE_CLOUD_PROJECT — setdefault with a stub value → downgraded
      to TODO with the source string preserved in the comment
    * MODEL — hardcoded model literal → replaced with os.getenv
    """
    recipe = root / "my-recipe"
    pkg = recipe / "my_recipe"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "agent.py").write_text(
        "import os\n"
        "os.environ.setdefault('USE_VERTEX', 'TRUE')\n"
        "os.environ.setdefault('GOOGLE_CLOUD_PROJECT', 'my-project-id')\n"
        'KEY = os.getenv("MY_API_KEY")\n'
        'LEVEL = os.getenv("LOG_LEVEL", "INFO")\n'
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

    # Simulate an existing .env.example with:
    #   * a v1-shape TODO line for USE_VERTEX — should be upgraded to TRUE
    #     (source has setdefault("USE_VERTEX", "TRUE"))
    #   * a v1-shape TODO for GOOGLE_CLOUD_PROJECT — should NOT be upgraded
    #     (source's setdefault value is placeholder-shaped "my-project-id")
    #   * a user-owned real value for LOG_LEVEL — should NOT be touched
    #     even though source has getenv("LOG_LEVEL", "INFO")
    (recipe / ".env.example").write_text(
        f"USE_VERTEX={m.PLACEHOLDER}\n"
        f"GOOGLE_CLOUD_PROJECT={m.PLACEHOLDER}\n"
        "LOG_LEVEL=WARN  # user set this deliberately\n",
        encoding="utf-8",
    )

    _run_main(monkeypatch, recipe)

    env_text = (recipe / ".env.example").read_text(encoding="utf-8")

    # v2.1 end-to-end assertions covering the pre-existing entry cases:
    #
    # 1. USE_VERTEX was a v1 TODO in .env.example; source has a real
    #    setdefault value → upgraded in place.
    assert "USE_VERTEX=TRUE" in env_text
    assert f"USE_VERTEX={m.PLACEHOLDER}" not in env_text
    # 2. GOOGLE_CLOUD_PROJECT was a v1 TODO; source has a placeholder-shape
    #    setdefault value → NOT upgraded (would be TODO → TODO churn).
    assert f"GOOGLE_CLOUD_PROJECT={m.PLACEHOLDER}" in env_text
    # 3. LOG_LEVEL had a user-typed real value + comment → USER_OWNED,
    #    never touched, even though source has getenv("LOG_LEVEL", "INFO").
    assert "LOG_LEVEL=WARN  # user set this deliberately" in env_text
    assert "LOG_LEVEL=INFO" not in env_text
    # 4. Bare os.getenv (no default) for a new var → TODO placeholder
    #    with the "no default in source" marker comment.
    assert f"MY_API_KEY={m.PLACEHOLDER}" in env_text
    # 5. Hardcoded model literal → env var + value written to .env.example.
    assert "MODEL_NAME=gemini-3.5-flash" in env_text

    # load_dotenv injected, python-dotenv added, model string replaced.
    assert "load_dotenv" in init_py.read_text(encoding="utf-8")
    assert "python-dotenv" in pyproject.read_text(encoding="utf-8")
    agent_text = agent_py.read_text(encoding="utf-8")
    # Rule 2: the replacement is BARE os.getenv("MODEL_NAME") — no inferred
    # default is written into Python source, even though the model string
    # is known and correct.
    assert 'os.getenv("MODEL_NAME")' in agent_text
    assert 'os.getenv("MODEL_NAME", ' not in agent_text
    assert "gemini-3.5-flash" not in agent_text
    # Rule 2: pre-existing setdefault calls are LEFT UNTOUCHED — we read
    # them but don't rewrite them.
    assert "os.environ.setdefault('USE_VERTEX', 'TRUE')" in agent_text
    assert (
        "os.environ.setdefault('GOOGLE_CLOUD_PROJECT', 'my-project-id')"
        in agent_text
    )


# ---------------------------------------------------------------------------
# End-to-end edge cases — main() against unusual recipe shapes
# ---------------------------------------------------------------------------


def test_main_recipe_with_no_python_files_does_not_crash(tmp_path, monkeypatch):
    # A recipe directory that has a pyproject.toml but no Python files
    # (empty scaffold, or someone deleted the source). Skill must
    # complete gracefully.
    recipe = tmp_path / "empty-recipe"
    recipe.mkdir()
    (recipe / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["python-dotenv>=1.0.0"]\n',
        encoding="utf-8",
    )

    # No Python files at all; no crash, no .env.example created (nothing
    # to write about).
    _run_main(monkeypatch, recipe)

    # .env.example is NOT created (there were no vars to write).
    assert not (recipe / ".env.example").exists()


def test_main_recipe_with_only_user_owned_env_example_is_noop(
    tmp_path, monkeypatch
):
    # If the maintainer has hand-authored every entry (no markers, no
    # v1 TODOs), the skill must NOT add or upgrade anything on that
    # file — and the file must stay byte-identical.
    recipe = tmp_path / "recipe"
    pkg = recipe / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "agent.py").write_text(
        "import os\n"
        "A = os.getenv('MY_API_KEY', 'default-A')\n"
        "B = os.getenv('MY_LOG_LEVEL', 'DEBUG')\n",
        encoding="utf-8",
    )
    (recipe / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["python-dotenv>=1.0.0"]\n',
        encoding="utf-8",
    )
    env_example = recipe / ".env.example"
    env_example.write_text(
        "# Fully hand-curated by maintainer\n"
        "MY_API_KEY=hand-typed-real-value\n"
        "MY_LOG_LEVEL=INFO  # user override\n",
        encoding="utf-8",
    )
    before = env_example.read_bytes()

    _run_main(monkeypatch, recipe)

    # Byte-identical: nothing was added, nothing was rewritten.
    assert env_example.read_bytes() == before


def test_main_recipe_with_unparseable_env_example_line_does_not_crash(
    tmp_path, monkeypatch
):
    # A .env.example with a malformed / non-declaration line should be
    # tolerated — the parser skips such lines and the skill continues.
    recipe = tmp_path / "recipe"
    pkg = recipe / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "agent.py").write_text(
        "import os\nKEY = os.getenv('NEW_VAR')\n",
        encoding="utf-8",
    )
    (recipe / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["python-dotenv>=1.0.0"]\n',
        encoding="utf-8",
    )
    (recipe / ".env.example").write_text(
        "GOOD_VAR=1\n"
        "this line is not a valid env declaration at all\n"
        "ALSO_GOOD=2\n",
        encoding="utf-8",
    )

    # Must complete without raising.
    _run_main(monkeypatch, recipe)

    # The garbage line is preserved verbatim (nothing to do with it).
    content = (recipe / ".env.example").read_text(encoding="utf-8")
    assert "this line is not a valid env declaration at all" in content
    # And NEW_VAR was appended normally.
    assert "NEW_VAR=" in content


def test_main_second_run_after_migration_is_byte_identical(
    tmp_path, monkeypatch
):
    # Cross-cutting idempotency: a full run that DOES modify things,
    # followed by an identical second run, must produce zero further
    # modifications. This is the strongest end-to-end guarantee that
    # the skill converges.
    recipe = _make_sample_recipe(tmp_path)
    (recipe / ".env.example").write_text(
        f"USE_VERTEX={m.PLACEHOLDER}\n"
        f"GOOGLE_CLOUD_PROJECT={m.PLACEHOLDER}\n"
        "LOG_LEVEL=WARN  # user set this deliberately\n",
        encoding="utf-8",
    )

    _run_main(monkeypatch, recipe)
    after_first_run = (recipe / ".env.example").read_bytes()

    _run_main(monkeypatch, recipe)
    after_second_run = (recipe / ".env.example").read_bytes()

    assert after_first_run == after_second_run


# ---------------------------------------------------------------------------
# Invariant / property-style tests — cut across many code paths at once,
# guarding the golden rule: user data never mutates, idempotency is total.
# ---------------------------------------------------------------------------


def test_invariant_user_owned_lines_byte_identical_after_any_writer_run(
    tmp_path,
):
    # Sweep across many scenarios (var missing / v1_todo / skill_todo /
    # skill_real / user_owned x source has real value / no default /
    # placeholder-shape default). At the end of ALL of them, every line
    # we classified as USER_OWNED must be byte-identical to what it
    # was before.
    env = _write(
        tmp_path / ".env.example",
        "# hand-tuned by maintainer\n"
        "USER_A=real-value-A\n"
        "USER_B=real-value-B  # inline note\n"
        "\texport USER_C=real-value-C\n"
        f"USER_D={m.PLACEHOLDER}  # touched-by-hand, keep as TODO\n"  # V1 TODO + comment → USER_OWNED
        "\n"
        "# section 2\n"
        "USER_E=user-value-E\n",
    )
    # Snapshot every line before any run.
    before_lines = env.read_text(encoding="utf-8").splitlines(keepends=True)

    # Run a mix of scenarios in one call: missing (append), pre-existing
    # SKILL_TODO getting upgraded, USER_OWNED with source drift.
    m.update_env_example(
        env,
        {
            # Missing → append (does not touch existing lines).
            "APPEND_ME": _srcs(("setdefault", "TRUE")),
            # Every USER_* var also appears in the source scan with a
            # different value than what the maintainer typed. Golden
            # rule: skill must never touch these.
            "USER_A": _srcs(("setdefault", "would-clobber")),
            "USER_B": _srcs(("setdefault", "would-clobber")),
            "USER_C": _srcs(("setdefault", "would-clobber")),
            "USER_D": _srcs(("setdefault", "would-clobber")),
            "USER_E": _srcs(("setdefault", "would-clobber")),
        },
    )

    after_lines = env.read_text(encoding="utf-8").splitlines(keepends=True)
    # Every ORIGINAL line must still be present, byte-identical.
    for original_line in before_lines:
        assert original_line in after_lines, (
            f"original user-owned line not preserved: {original_line!r}"
        )


def test_invariant_idempotency_across_all_v2_paths(tmp_path):
    # A single .env.example exercising append, upgrade-v1, upgrade-skill,
    # skip-user-owned, skip-placeholder-source, skip-drifted-skill-real
    # in one shot. Run the writer twice; second call must return
    # ([], []) and file must be byte-identical.
    env = _write(
        tmp_path / ".env.example",
        f"V1_TODO_UPGRADABLE={m.PLACEHOLDER}\n"
        f"V1_TODO_NO_SOURCE={m.PLACEHOLDER}\n"  # source has no default → skip
        f"V1_TODO_PLACEHOLDER_SOURCE={m.PLACEHOLDER}\n"  # source has "my-x" → skip
        f"SKILL_TODO_UPGRADABLE={m.PLACEHOLDER}"
        f"  # {m.EXTRACTED_MARKER}; no default in source\n"
        f"SKILL_REAL_DRIFTED=old-value"
        f"  # {m.EXTRACTED_MARKER}; from setdefault in x.py\n"
        "USER_OWNED_VAR=user-value\n",
    )

    env_vars = {
        "V1_TODO_UPGRADABLE": _srcs(("setdefault", "real")),
        "V1_TODO_NO_SOURCE": _srcs(("getenv", None)),
        "V1_TODO_PLACEHOLDER_SOURCE": _srcs(("setdefault", "my-thing")),
        "SKILL_TODO_UPGRADABLE": _srcs(("setdefault", "TRUE")),
        "SKILL_REAL_DRIFTED": _srcs(("setdefault", "new-drifted-value")),
        "USER_OWNED_VAR": _srcs(("setdefault", "would-clobber")),
        "APPEND_ME": _srcs(("setdefault", "hello")),
    }

    added1, upgraded1 = m.update_env_example(env, env_vars)
    content_after_first = env.read_bytes()

    added2, upgraded2 = m.update_env_example(env, env_vars)
    content_after_second = env.read_bytes()

    # First run: added APPEND_ME; upgraded the two upgradable TODOs.
    assert set(added1) == {"APPEND_ME"}
    assert set(upgraded1) == {
        "V1_TODO_UPGRADABLE",
        "SKILL_TODO_UPGRADABLE",
    }
    # Second run: everything already in its final state.
    assert added2 == []
    assert upgraded2 == []
    assert content_after_first == content_after_second


def test_invariant_writer_never_produces_unclassifiable_entries(tmp_path):
    # Every line the writer touches must re-parse cleanly into one of
    # the four EntryClassification buckets. This guards against a
    # malformed line being accidentally emitted (which would then
    # confuse the classifier on the next run).
    env = _write(
        tmp_path / ".env.example",
        f"V1_TODO={m.PLACEHOLDER}\n",
    )

    m.update_env_example(
        env,
        {
            "V1_TODO": _srcs(("setdefault", "real")),
            "APPEND_ME": _srcs(("getenv", "value")),
            "APPEND_TODO": _srcs(("getenv", None)),
            "APPEND_DOWNGRADED": _srcs(("setdefault", "my-project-id")),
        },
    )

    # Every entry the writer produced must classify as one of the
    # known buckets. Nothing should be "unclassifiable" (which would
    # be caught by the round-trip check but is worth belt-and-braces
    # testing at the invariant level).
    valid_buckets = {
        m.EntryClassification.USER_OWNED,
        m.EntryClassification.V1_TODO,
        m.EntryClassification.SKILL_TODO,
        m.EntryClassification.SKILL_REAL,
    }
    entries = m.parse_env_example(env)
    for name, entry in entries.items():
        assert entry.classification in valid_buckets, (
            f"{name} classified as unknown bucket: {entry.classification}"
        )


def test_invariant_appended_entries_are_skill_owned(tmp_path):
    # Every entry APPENDED by the writer must be either SKILL_TODO or
    # SKILL_REAL — never USER_OWNED (which would mean the skill's own
    # writes don't carry the marker properly). This closes the loop
    # so future runs can classify + upgrade correctly.
    env = tmp_path / ".env.example"

    m.update_env_example(
        env,
        {
            "WITH_DEFAULT": _srcs(("setdefault", "real")),
            "NO_DEFAULT": _srcs(("getenv", None)),
            "DOWNGRADED": _srcs(("setdefault", "your-secret")),
        },
    )

    entries = m.parse_env_example(env)
    assert (
        entries["WITH_DEFAULT"].classification
        == m.EntryClassification.SKILL_REAL
    )
    assert (
        entries["NO_DEFAULT"].classification == m.EntryClassification.SKILL_TODO
    )
    assert (
        entries["DOWNGRADED"].classification == m.EntryClassification.SKILL_TODO
    )


# ---------------------------------------------------------------------------
# _post_header_index — module-docstring detection
#
# Regression: the docstring scan was a bare startswith('"""') test, so any
# legal string prefix (r"""..., u'''...) slipped past it and `import os` was
# injected ABOVE the docstring. That demotes the docstring to a dead string
# expression and loses __doc__.
# ---------------------------------------------------------------------------


def _insert_point(source: str) -> str:
    """Return the line `import os` would be inserted before."""
    lines = source.splitlines(keepends=True)
    idx = m._post_header_index(lines)
    return lines[idx] if idx < len(lines) else "<EOF>"


def test_post_header_index_skips_plain_docstring():
    src = '# license\n\n"""Doc.\n\nMore.\n"""\n\nimport sys\n'
    assert _insert_point(src) == "\n"


def test_post_header_index_skips_raw_prefixed_docstring():
    src = '# license\n\nr"""Doc.\n\nMore.\n"""\n\nimport sys\n'
    assert _insert_point(src) == "\n"


def test_post_header_index_skips_unicode_prefixed_docstring():
    src = '# license\n\nu"""Doc.\n"""\n\nimport sys\n'
    assert _insert_point(src) == "\n"


def test_post_header_index_skips_raw_single_quoted_docstring():
    src = "# license\n\nr'''Doc.\nMore.\n'''\n\nimport sys\n"
    assert _insert_point(src) == "\n"


def test_post_header_index_skips_one_line_raw_docstring():
    src = '# license\n\nr"""Doc."""\n\nimport sys\n'
    assert _insert_point(src) == "\n"


def test_post_header_index_handles_shebang_and_raw_docstring():
    src = '#!/usr/bin/env python3\n# license\n\nr"""Doc.\n"""\n\nimport sys\n'
    assert _insert_point(src) == "\n"


def test_post_header_index_without_docstring_lands_after_comments():
    src = "# license\n\nimport sys\n"
    assert _insert_point(src) == "import sys\n"


def test_post_header_index_unparseable_source_falls_back_to_scan():
    # The AST path is unavailable, but the textual fallback must still
    # respect a prefixed docstring.
    src = '# license\n\nr"""Doc.\n"""\n\nimport sys\ndef (\n'
    assert _insert_point(src) == "\n"


def test_post_header_index_empty_file():
    assert m._post_header_index([]) == 0


def test_os_import_injected_below_raw_docstring(tmp_path):
    # End-to-end: the module docstring must remain the first statement, so
    # ast.get_docstring() still finds it after the rewrite.
    src = (
        "# Copyright\n"
        "\n"
        'r"""Module docstring with a \\d regex mention."""\n'
        "\n"
        'DEFAULT_MODEL = "gemini-embedding-001"\n'
    )
    py = _write(tmp_path / "ingest.py", src)

    hits = m.extract_hardcoded_models([py])
    name_map = m.assign_model_var_names({"gemini-embedding-001"})
    m.replace_hardcoded_models([py], hits, name_map)

    result = py.read_text(encoding="utf-8")
    tree = ast.parse(result)
    assert ast.get_docstring(tree) is not None, (
        f"docstring lost — import injected above it:\n{result}"
    )
    assert "import os" in result


# ---------------------------------------------------------------------------
# _insert_before_close — trailing comma must land on the entry, not a comment
# ---------------------------------------------------------------------------


def _deps_after_dotenv(body: str):
    """Add python-dotenv to a dependencies array and parse the result."""
    src = '[project]\nname = "x"\n' + body + "\n"
    out = m._compute_pyproject_with_dotenv(src)
    assert out is not None
    return out, m.tomllib.loads(out)["project"]["dependencies"]


def test_insert_dotenv_preserves_comment_when_entry_has_comma():
    out, deps = _deps_after_dotenv(
        'dependencies = [\n    "requests>=2.28",\n'
        '    "pyOpenSSL>=23.0",  # mTLS during auth\n]'
    )
    assert deps == ["requests>=2.28", "pyOpenSSL>=23.0", "python-dotenv>=1.0.0"]
    # The comment must be untouched — no comma appended inside it.
    assert "# mTLS during auth\n" in out
    assert "# mTLS during auth," not in out


def test_insert_dotenv_adds_comma_before_trailing_comment():
    # Previously produced invalid TOML: the separating comma was written
    # inside the comment, so the array never closed.
    out, deps = _deps_after_dotenv(
        'dependencies = [\n    "requests>=2.28",\n'
        '    "pyOpenSSL>=23.0"  # no comma here\n]'
    )
    assert deps == ["requests>=2.28", "pyOpenSSL>=23.0", "python-dotenv>=1.0.0"]
    assert '"pyOpenSSL>=23.0",  # no comma here' in out


def test_insert_dotenv_with_comment_only_last_line():
    _out, deps = _deps_after_dotenv(
        'dependencies = [\n    "requests>=2.28",\n    # a note\n]'
    )
    assert deps == ["requests>=2.28", "python-dotenv>=1.0.0"]


def test_insert_dotenv_into_empty_array():
    _out, deps = _deps_after_dotenv("dependencies = [\n]")
    assert deps == ["python-dotenv>=1.0.0"]


def test_insert_dotenv_into_empty_array_with_comment():
    _out, deps = _deps_after_dotenv("dependencies = [\n    # nothing yet\n]")
    assert deps == ["python-dotenv>=1.0.0"]


def test_insert_dotenv_into_single_line_array():
    _out, deps = _deps_after_dotenv('dependencies = ["a", "b"]')
    assert deps == ["a", "b", "python-dotenv>=1.0.0"]


def test_insert_dotenv_ignores_hash_inside_string():
    # A '#' inside a PEP 508 URL is not a comment marker.
    _out, deps = _deps_after_dotenv(
        'dependencies = [\n    "pkg @ https://x/y#egg=pkg"\n]'
    )
    assert deps == ["pkg @ https://x/y#egg=pkg", "python-dotenv>=1.0.0"]


def test_split_trailing_comment_respects_quotes():
    assert m._split_trailing_comment('"a#b"  # real') == (
        '"a#b"  ',
        "# real",
    )
    assert m._split_trailing_comment("'a#b'") == ("'a#b'", "")
    assert m._split_trailing_comment('"esc\\"#q"  # c') == (
        '"esc\\"#q"  ',
        "# c",
    )


# ---------------------------------------------------------------------------
# Model rewrite: bare getenv only when a load_dotenv bootstrap exists
# ---------------------------------------------------------------------------


def test_model_rewrite_keeps_default_when_no_dotenv_bootstrap(tmp_path):
    py = _write(
        tmp_path / "ingest.py",
        'DEFAULT_MODEL = "gemini-embedding-001"\n',
    )
    hits = m.extract_hardcoded_models([py])
    name_map = {"gemini-embedding-001": "EMBEDDING_MODEL"}

    m.replace_hardcoded_models([py], hits, name_map, keep_defaults=True)

    result = py.read_text(encoding="utf-8")
    assert 'os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")' in result
    # Still valid Python, and the value can never be None.
    ast.parse(result)


def test_model_rewrite_bare_getenv_by_default(tmp_path):
    py = _write(
        tmp_path / "agent.py",
        'MODEL = "gemini-3.5-flash"\n',
    )
    hits = m.extract_hardcoded_models([py])
    name_map = {"gemini-3.5-flash": "MODEL_NAME"}

    m.replace_hardcoded_models([py], hits, name_map)

    result = py.read_text(encoding="utf-8")
    assert 'os.getenv("MODEL_NAME")' in result
    assert "gemini-3.5-flash" not in result


def test_run_step_load_dotenv_returns_false_without_package(tmp_path, capsys):
    assert m.run_step_load_dotenv(tmp_path) is False
    assert "No Python package" in capsys.readouterr().out


def test_run_step_load_dotenv_returns_true_with_package(tmp_path):
    _write(tmp_path / "app" / "__init__.py", "")
    _write(tmp_path / "app" / "agent.py", "")
    assert m.run_step_load_dotenv(tmp_path) is True


def test_py_string_literal_escapes():
    assert m._py_string_literal("plain") == '"plain"'
    assert m._py_string_literal('has"quote') == '"has\\"quote"'
    assert m._py_string_literal("back\\slash") == '"back\\\\slash"'
    assert m._py_string_literal("nl\n") == '"nl\\n"'


# ---------------------------------------------------------------------------
# Model var naming derived from the assignment target
# ---------------------------------------------------------------------------


def test_model_var_hint_from_default_prefixed_constant(tmp_path):
    py = _write(
        tmp_path / "ingest.py",
        'DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"\n',
    )
    assert m.extract_model_var_hints([py]) == {
        "gemini-embedding-001": "EMBEDDING_MODEL"
    }


def test_model_var_hint_from_nested_call_value(tmp_path):
    py = _write(
        tmp_path / "setup.py",
        'embedding_model = cfg.get("embedding_model", "gemini-embedding-001")\n',
    )
    assert m.extract_model_var_hints([py]) == {
        "gemini-embedding-001": "EMBEDDING_MODEL"
    }


def test_model_var_hint_ignores_target_without_model_in_name(tmp_path):
    py = _write(tmp_path / "agent.py", 'LLM = "gemini-3.5-flash"\n')
    assert m.extract_model_var_hints([py]) == {}


def test_model_var_hint_bare_model_target_maps_to_model_name(tmp_path):
    py = _write(tmp_path / "agent.py", 'MODEL = "gemini-3.5-flash"\n')
    assert m.extract_model_var_hints([py]) == {"gemini-3.5-flash": "MODEL_NAME"}


def test_model_var_hint_ignores_docstring_mention(tmp_path):
    py = _write(
        tmp_path / "agent.py",
        '"""Uses gemini-3.5-flash by default."""\n',
    )
    assert m.extract_model_var_hints([py]) == {}


def test_model_var_hint_deterministic_across_conflicting_targets(tmp_path):
    a = _write(tmp_path / "a.py", 'EMBEDDING_MODEL = "gemini-embedding-001"\n')
    b = _write(tmp_path / "b.py", 'VECTOR_MODEL = "gemini-embedding-001"\n')
    c = _write(tmp_path / "c.py", 'VECTOR_MODEL = "gemini-embedding-001"\n')
    # VECTOR_MODEL has two votes to EMBEDDING_MODEL's one.
    assert m.extract_model_var_hints([a, b, c]) == {
        "gemini-embedding-001": "VECTOR_MODEL"
    }


def test_assign_model_var_names_uses_hint():
    mapping = m.assign_model_var_names(
        {"gemini-embedding-001"},
        existing_vars={"GEMINI_MODEL"},
        hints={"gemini-embedding-001": "EMBEDDING_MODEL"},
    )
    assert mapping == {"gemini-embedding-001": "EMBEDDING_MODEL"}


def test_assign_model_var_names_falls_back_when_hint_taken():
    mapping = m.assign_model_var_names(
        {"gemini-embedding-001"},
        existing_vars={"EMBEDDING_MODEL"},
        hints={"gemini-embedding-001": "EMBEDDING_MODEL"},
    )
    assert mapping == {"gemini-embedding-001": "MODEL_NAME"}


def test_assign_model_var_names_mixed_hinted_and_unhinted():
    mapping = m.assign_model_var_names(
        {"gemini-embedding-001", "gemini-3.5-flash"},
        hints={"gemini-embedding-001": "EMBEDDING_MODEL"},
    )
    assert mapping["gemini-embedding-001"] == "EMBEDDING_MODEL"
    # The lone remaining model still gets the unsuffixed generic name.
    assert mapping["gemini-3.5-flash"] == "MODEL_NAME"


def test_assign_model_var_names_unchanged_without_hints():
    # Backwards compatibility with the pre-hint behaviour.
    assert m.assign_model_var_names({"gemini-3.5-flash"}) == {
        "gemini-3.5-flash": "MODEL_NAME"
    }
    two = m.assign_model_var_names({"gemini-a", "gemini-b"})
    assert set(two.values()) == {
        "MODEL_NAME_GENERATED_1",
        "MODEL_NAME_GENERATED_2",
    }


# ---------------------------------------------------------------------------
# Structural exclusions: position decides whether a model literal is a
# configurable constant.
#
# Regression: the extractor rewrote every gemini-* string it found, including
# the keys AND values of a model lookup table:
#
#     IMAGE_MODELS = {"flash": "gemini-2.5-flash-image", ...}
#
# Turning a dict key into os.getenv(...) means IMAGE_MODELS.get("flash")
# never matches again — a silent, total break of the lookup.
# ---------------------------------------------------------------------------


LOOKUP_TABLE_SRC = """IMAGE_MODELS = {
    "gemini-2.5-flash-image": "gemini-2.5-flash-image",
    "flash": "gemini-2.5-flash-image",
    "pro": "gemini-2.5-pro-image",
}


def resolve(label):
    if "gemini-3.1" in label:
        return IMAGE_MODELS["gemini-2.5-pro-image"]
    return IMAGE_MODELS.get(label, IMAGE_MODELS["gemini-2.5-flash-image"])
"""


def test_dict_literal_entries_are_not_detected(tmp_path):
    py = _write(tmp_path / "models.py", LOOKUP_TABLE_SRC)
    assert m.extract_hardcoded_models([py]) == {}


def test_dict_literal_is_left_byte_identical(tmp_path):
    py = _write(tmp_path / "models.py", LOOKUP_TABLE_SRC)
    hits = m.extract_hardcoded_models([py])
    name_map = m.assign_model_var_names(
        {"gemini-2.5-flash-image", "gemini-2.5-pro-image", "gemini-3.1"}
    )

    m.replace_hardcoded_models([py], hits, name_map)

    assert py.read_text(encoding="utf-8") == LOOKUP_TABLE_SRC


def test_list_set_and_tuple_entries_are_excluded(tmp_path):
    py = _write(
        tmp_path / "c.py",
        'ALLOWED = ["gemini-3.5-flash"]\n'
        'SUPPORTED = {"gemini-3.5-pro"}\n'
        'PAIR = ("gemini-3.1-flash-image",)\n',
    )
    assert m.extract_hardcoded_models([py]) == {}


def test_subscript_index_is_excluded(tmp_path):
    py = _write(tmp_path / "c.py", 'X = TABLE["gemini-3.5-flash"]\n')
    assert m.extract_hardcoded_models([py]) == {}


def test_comparison_operand_is_excluded(tmp_path):
    py = _write(
        tmp_path / "c.py",
        'if "gemini-3.1" in model_id:\n    pass\n',
    )
    assert m.extract_hardcoded_models([py]) == {}


def test_scalar_assignment_is_still_detected(tmp_path):
    # The exclusions must not swallow the case the skill exists for.
    py = _write(tmp_path / "agent.py", 'MODEL = "gemini-3.5-flash"\n')
    hits = m.extract_hardcoded_models([py])
    assert [v for hits_ in hits.values() for _, v in hits_] == [
        "gemini-3.5-flash"
    ]


def test_call_keyword_argument_is_still_detected(tmp_path):
    py = _write(tmp_path / "agent.py", 'a = Agent(model="gemini-3.5-flash")\n')
    hits = m.extract_hardcoded_models([py])
    assert [v for hits_ in hits.values() for _, v in hits_] == [
        "gemini-3.5-flash"
    ]


def test_dict_target_does_not_produce_a_var_name_hint(tmp_path):
    # `IMAGE_MODELS = {...}` names a TABLE, not a model. Deriving the env var
    # name IMAGE_MODELS from it produced a nonsensical .env.example entry.
    py = _write(tmp_path / "models.py", LOOKUP_TABLE_SRC)
    assert m.extract_model_var_hints([py]) == {}


def test_skipped_literals_are_reported(tmp_path):
    py = _write(tmp_path / "models.py", LOOKUP_TABLE_SRC)
    skipped = m.extract_skipped_model_literals([py])

    values = [v for _, v in skipped[py]]
    assert "gemini-2.5-flash-image" in values  # dict key + value
    assert "gemini-2.5-pro-image" in values
    assert "gemini-3.1" in values  # comparison operand


def test_skipped_literals_deduplicate_per_line(tmp_path):
    # `"x": "x"` is two nodes on one line; report it once.
    py = _write(
        tmp_path / "d.py",
        'T = {\n    "gemini-3.5-flash": "gemini-3.5-flash",\n}\n',
    )
    skipped = m.extract_skipped_model_literals([py])
    assert skipped[py] == [(2, "gemini-3.5-flash")]


def test_skipped_literals_ignore_docstring_mentions(tmp_path):
    py = _write(
        tmp_path / "d.py",
        '"""Uses gemini-3.5-flash by default."""\n',
    )
    assert m.extract_skipped_model_literals([py]) == {}


def test_skipped_report_empty_when_nothing_excluded(tmp_path):
    py = _write(tmp_path / "agent.py", 'MODEL = "gemini-3.5-flash"\n')
    assert m.extract_skipped_model_literals([py]) == {}
