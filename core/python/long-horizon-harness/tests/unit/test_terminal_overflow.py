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

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from horizon.environment import LocalEnvironment
from horizon.environment_context import (
    clear_active_environment,
    set_active_environment,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def env_root(tmp_path: Path) -> Iterator[Path]:
    working_dir = tmp_path / "ws"
    working_dir.mkdir()
    set_active_environment(LocalEnvironment(working_dir=working_dir))
    prev = os.getcwd()
    os.chdir(working_dir)
    try:
        yield working_dir
    finally:
        os.chdir(prev)
        clear_active_environment()


async def test_small_output_is_verbatim_no_pointer(env_root: Path) -> None:
    from horizon.tools.terminal_exec import terminal

    result = await terminal(command="echo hello")
    assert result["stdout"].strip() == "hello"
    assert result["truncated"] is False
    assert "stdout_overflow_path" not in result


async def test_large_stdout_spills_and_returns_pointer(env_root: Path) -> None:
    from horizon.tools.terminal_exec import terminal

    # Produce > 50 KB on stdout.
    result = await terminal(
        command="for i in $(seq 1 20000); do echo line-$i; done"
    )
    assert result["truncated"] is True
    assert "Full output saved to" in result["stdout"]
    assert "view_file with offset/limit" in result["stdout"]
    assert "stdout_overflow_path" in result
    # The spilled file holds the FULL output and is readable via the env interface.
    saved = Path(result["stdout_overflow_path"]).read_text(encoding="utf-8")
    assert "line-20000" in saved
    assert "line-1\n" in saved


async def test_timeout_kill_path_includes_retry_hint(env_root: Path) -> None:
    from horizon.tools.terminal_exec import terminal

    result = await terminal(command="sleep 5", timeout_s=1)
    assert result["timed_out"] is True
    assert "hint" in result
    assert "timeout_s" in result["hint"]
