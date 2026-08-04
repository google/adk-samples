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

"""Oversized ``terminal()`` output spills to ``lha/tool-output/`` via the interface.

Direct tool-call rather than LLM-driven — same pattern as
``test_terminal_tool.py``. The smoke contract is "tool dispatch + overflow
spill + env wiring works through the interface"; LLM behavior is covered by
``tests/eval``. The spilled file is read back through the active env so the
round-trip is exercised on both the local and sandbox backends.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk.environment import BaseEnvironment

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_large_output_spills_to_tool_output_dir(
    active_env: BaseEnvironment,
) -> None:
    from horizon.tools.terminal_exec import terminal

    # ~600 KB on stdout — well past the 50 KB / 800-line cap.
    marker = "OVERFLOW_SMOKE_MARKER"
    command = f"python3 -c \"print('\\n'.join('{marker}-%05d' % i for i in range(30000)))\""
    result = await terminal(command=command)

    assert result["exit_code"] == 0
    assert result["truncated"] is True
    assert "Full output saved to" in result["stdout"]
    assert "view_file with offset/limit" in result["stdout"]
    assert "stdout_overflow_path" in result

    spill_path = Path(result["stdout_overflow_path"])
    root = active_env.working_dir.resolve()
    assert spill_path.parent == (root / "lha" / "tool-output").resolve()
    assert spill_path.is_relative_to(root)

    # FULL output is recoverable through the env interface, nothing dropped.
    raw = await active_env.read_file(spill_path)
    saved = raw.decode("utf-8")
    assert f"{marker}-00000" in saved
    assert f"{marker}-29999" in saved


async def test_small_output_does_not_spill(active_env: BaseEnvironment) -> None:
    from horizon.tools.terminal_exec import terminal

    result = await terminal(command="echo small-output-marker")

    assert result["exit_code"] == 0
    assert result["truncated"] is False
    assert "small-output-marker" in result["stdout"]
    assert "Full output saved to" not in result["stdout"]
    assert "stdout_overflow_path" not in result
    assert "stderr_overflow_path" not in result


async def test_agent_facing_wrapper_spills_on_default_path(
    active_env: BaseEnvironment,
) -> None:
    # The agent calls the background-aware wrapper, whose default
    # on_timeout='background' path must also spill (not just the foreground tool).
    from horizon.tools.processes.terminal import terminal as wrapper_terminal

    marker = "WRAPPER_OVERFLOW_MARKER"
    command = f"python3 -c \"print('\\n'.join('{marker}-%05d' % i for i in range(30000)))\""
    result = await wrapper_terminal(command=command)

    assert result["truncated"] is True
    assert "session_id" not in result  # completed foreground, not promoted
    assert "stdout_overflow_path" in result

    spill_path = Path(result["stdout_overflow_path"])
    root = active_env.working_dir.resolve()
    assert spill_path.parent == (root / "lha" / "tool-output").resolve()

    saved = (await active_env.read_file(spill_path)).decode("utf-8")
    assert f"{marker}-29999" in saved
