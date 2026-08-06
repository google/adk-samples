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
"""Unit tests for check_env_vars.py.

The undeclared-variable error is the highest-volume CI failure in this repo,
so these tests pin what the message has to contain — the reading site, a
literal line to paste, and the skill that does it for you — as well as the
exit-code contract the workflow branches on.
"""

import sys
from pathlib import Path

import check_env_vars as m

# The numbers the workflow branches on. Spelled out rather than imported so
# a change to them has to be made deliberately in two places.
EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_CI_FAULT = 2


def _recipe(tmp_path: Path, env_example: str | bytes | None, **sources: str):
    if env_example is not None:
        target = tmp_path / ".env.example"
        if isinstance(env_example, bytes):
            target.write_bytes(env_example)
        else:
            target.write_text(env_example, encoding="utf-8")
    for name, body in sources.items():
        path = tmp_path / f"{name}.py"
        path.write_text(body, encoding="utf-8")
    return tmp_path


def _run(tmp_path: Path, monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", ["check_env_vars.py", str(tmp_path)])
    return m.main()


def test_declared_variable_passes(tmp_path, monkeypatch, capsys):
    _recipe(
        tmp_path,
        "GOOGLE_CLOUD_PROJECT=my-project\n",
        agent='import os\n\nP = os.getenv("GOOGLE_CLOUD_PROJECT")\n',
    )
    assert _run(tmp_path, monkeypatch) == EXIT_OK
    assert "::error" not in capsys.readouterr().out


def test_allowlisted_variable_is_not_reported(tmp_path, monkeypatch):
    _recipe(
        tmp_path,
        "GOOGLE_CLOUD_PROJECT=my-project\n",
        agent='import os\n\nH = os.environ["HOME"]\n',
    )
    assert _run(tmp_path, monkeypatch) == EXIT_OK


def test_undeclared_variable_names_the_file_and_line(
    tmp_path, monkeypatch, capsys
):
    _recipe(
        tmp_path,
        "GOOGLE_CLOUD_PROJECT=my-project\n",
        agent=(
            "import os\n"
            "\n"
            'P = os.getenv("GOOGLE_CLOUD_PROJECT")\n'
            "M = os.getenv(\n"
            '    "MODEL_NAME",\n'
            '    "gemini-3.5-flash",\n'
            ")\n"
        ),
    )
    assert _run(tmp_path, monkeypatch) == EXIT_VIOLATIONS
    out = capsys.readouterr().out
    assert "MODEL_NAME" in out
    # The read is on line 4 even though the string literal is on line 5.
    assert f"{tmp_path / 'agent.py'}:4" in out
    # A literal line to paste, not a description of one.
    assert f"MODEL_NAME={m.PLACEHOLDER}" in out
    # The skill the handbook tells contributors to use.
    assert "extract-python-environment-variables" in out
    assert f"::error file={tmp_path / '.env.example'}::" in out


def test_each_undeclared_variable_gets_its_own_annotation(
    tmp_path, monkeypatch, capsys
):
    _recipe(
        tmp_path,
        "# nothing declared\n",
        agent='import os\n\nA = os.getenv("ONE")\nB = os.getenv("TWO")\n',
    )
    assert _run(tmp_path, monkeypatch) == EXIT_VIOLATIONS
    assert capsys.readouterr().out.count("::error file=") == 2


def test_a_file_that_does_not_parse_is_reported_not_skipped(
    tmp_path, monkeypatch, capsys
):
    """A silent skip let a recipe with a syntax error report a false PASS:
    every env read inside the unparsed file was invisible to the check."""
    _recipe(tmp_path, "GOOGLE_CLOUD_PROJECT=x\n", broken="def oops(\n")
    assert _run(tmp_path, monkeypatch) == EXIT_VIOLATIONS
    out = capsys.readouterr().out
    assert "not valid Python" in out
    assert "[PASS]" not in out
    assert f"::error file={tmp_path / 'broken.py'}::" in out


def test_non_utf8_env_example_is_reported_not_crashed(
    tmp_path, monkeypatch, capsys
):
    _recipe(
        tmp_path,
        "MY_VAR=caf\u00e9\n".encode("latin-1"),
        agent='import os\n\nV = os.getenv("MY_VAR")\n',
    )
    assert _run(tmp_path, monkeypatch) == EXIT_VIOLATIONS
    out = capsys.readouterr().out
    assert "Traceback" not in out
    assert "not valid UTF-8" in out
    # Every variable would look undeclared against an unreadable file, so
    # the encoding is the only thing reported.
    assert out.count("::error file=") == 1


def test_missing_env_example_is_not_this_checkers_failure(
    tmp_path, monkeypatch, capsys
):
    _recipe(tmp_path, None, agent='import os\n\nV = os.getenv("X")\n')
    assert _run(tmp_path, monkeypatch) == EXIT_OK
    assert "::error" not in capsys.readouterr().out


def test_a_path_that_is_not_a_directory_is_a_ci_fault(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys, "argv", ["check_env_vars.py", str(tmp_path / "nope")]
    )
    assert m.main() == EXIT_CI_FAULT
    out = capsys.readouterr().out
    assert "[ci-fault]" in out
    assert "::error file=" not in out


def test_an_unexpected_crash_is_a_ci_fault_not_the_contributors_fault(
    tmp_path, monkeypatch, capsys
):
    def boom(*_args, **_kwargs):
        raise RuntimeError("checker bug")

    monkeypatch.setattr(m, "_collect_used_vars", boom)
    _recipe(tmp_path, "A=1\n")
    assert _run(tmp_path, monkeypatch) == EXIT_CI_FAULT
    out = capsys.readouterr().out
    assert "RuntimeError: checker bug" in out
    assert "::error file=" not in out
