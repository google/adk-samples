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

"""``terminal()`` routes through the shared environment.

Direct tool-call rather than LLM-driven — same pattern as the existing
``tests/integration/probes`` (see ``probe_cuj4_artifact.py``). The smoke
contract is "tool dispatch + env wiring works"; LLM behavior is covered
by ``tests/eval``.
"""

from __future__ import annotations

import pytest
from google.adk.environment import BaseEnvironment

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_echo_round_trips_through_env(
    active_env: BaseEnvironment,
) -> None:
    from horizon.tools.terminal_exec import terminal

    result = await terminal(command="echo smoke-marker")

    assert result["exit_code"] == 0
    assert "smoke-marker" in result["stdout"]
    assert result["timed_out"] is False


async def test_pwd_returns_env_working_dir(active_env: BaseEnvironment) -> None:
    from horizon.tools.terminal_exec import terminal

    result = await terminal(command="pwd")
    assert result["exit_code"] == 0
    assert str(active_env.working_dir.resolve()) in result["stdout"]


async def test_cwd_outside_working_dir_is_rejected(
    active_env: BaseEnvironment,
) -> None:
    from horizon.tools.terminal_exec import terminal

    result = await terminal(command="ls", cwd="/etc")
    assert result["exit_code"] == -1
    assert "outside working_dir" in result["stderr"]
