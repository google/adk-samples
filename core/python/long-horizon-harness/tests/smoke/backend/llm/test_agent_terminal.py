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

"""LLM-driven: agent picks ``terminal`` for shell-execution prompts.

Asserts on tool trajectory only — the prompt is shaped to make one tool
call obvious so we can lock in the wiring without depending on Gemini's
natural-language output.
"""

from __future__ import annotations

import pytest

from tests.smoke.backend.llm.conftest import function_call_names

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_agent_invokes_terminal_for_echo_prompt(llm_runner) -> None:
    events = await llm_runner.run_once(
        "Use the `terminal` tool to run exactly this command: echo smoke-llm-marker"
    )

    names = function_call_names(events)
    assert "terminal" in names, (
        f"expected agent to call `terminal`; got tool calls: {names}"
    )


async def test_agent_invokes_terminal_for_pwd_prompt(llm_runner) -> None:
    events = await llm_runner.run_once(
        "Call the `terminal` tool with the command `pwd` and report the result."
    )

    names = function_call_names(events)
    assert "terminal" in names, (
        f"expected agent to call `terminal`; got tool calls: {names}"
    )
