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

"""``terminal()`` (the LLM-callable wrapper) defaults cwd to
``active_environment().working_dir`` and rejects explicit cwds that
escape it.

``run_command`` (the lower-level helper) stays raw — these tests target
the wrapper only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.asyncio,
]


async def test_terminal_default_cwd_is_env_working_dir(tmp_path: Path) -> None:
    from horizon.tools.terminal_exec import terminal

    result = await terminal("pwd")

    produced = Path(result["stdout"].strip()).resolve()
    assert produced == tmp_path.resolve()


async def test_terminal_relative_cwd_resolves_under_env(tmp_path: Path) -> None:
    from horizon.tools.terminal_exec import terminal

    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "note.txt").write_text("hello")

    result = await terminal("cat note.txt", cwd="sub")

    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]


async def test_terminal_absolute_cwd_under_env_allowed(tmp_path: Path) -> None:
    from horizon.tools.terminal_exec import terminal

    (tmp_path / "deep").mkdir()
    result = await terminal("pwd", cwd=str(tmp_path / "deep"))

    assert result["exit_code"] == 0
    produced = Path(result["stdout"].strip()).resolve()
    assert produced == (tmp_path / "deep").resolve()


async def test_terminal_absolute_cwd_outside_env_rejected(
    tmp_path: Path,
) -> None:
    from horizon.tools.terminal_exec import terminal

    result = await terminal("pwd", cwd="/tmp")

    # No subprocess fired; structured rejection.
    assert result.get("exit_code", -1) != 0
    error = (result.get("stderr") or result.get("error") or "").lower()
    assert "outside" in error or "denied" in error


async def test_terminal_dotdot_cwd_escape_rejected(tmp_path: Path) -> None:
    from horizon.tools.terminal_exec import terminal

    result = await terminal("pwd", cwd="../..")

    assert result.get("exit_code", -1) != 0
    error = (result.get("stderr") or result.get("error") or "").lower()
    assert "outside" in error or "denied" in error
