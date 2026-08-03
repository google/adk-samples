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

"""LLM-driven: agent tracks a multi-step task with the write_todos tool.

A multi-step prompt should drive the agent to call ``write_todos``, which
persists one markdown file per item under ``lha/todos/`` through the env interface.
We assert the durable side effect (files on disk + parseable todos), not the
agent's wording. The second-turn "context includes the rendered block" claim
from the spec is validated by its durable proxy: the folder survives and
``read_todos`` parses it (the per-turn reminder assembly reads the same folder
every turn).
"""

from __future__ import annotations

import pytest
from google.adk.environment import BaseEnvironment

from tests.smoke.backend.llm.conftest import function_call_names

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_agent_persists_todos_for_multistep_task(
    llm_runner, active_env: BaseEnvironment
) -> None:
    prompt = (
        "I have a multi-step task. Please plan it out using your todo tool "
        "before starting: (1) create a file notes.txt, (2) write three "
        "lines into it, (3) verify the file has three lines. Record these "
        "as a todo list first."
    )

    events = await llm_runner.run_once(prompt)

    # Trajectory: the agent reached for write_todos.
    names = function_call_names(events)
    assert "write_todos" in names, (
        f"expected agent to call write_todos; got tool calls: {names}"
    )

    # Durable side effect: todo files persisted under lha/todos/ via the interface.
    todos_dir = active_env.working_dir.resolve() / "lha" / "todos"
    listing = await active_env.execute(f"ls -1 {todos_dir} 2>/dev/null || true")
    assert ".md" in listing.stdout, (
        f"expected todo files under {todos_dir}; ls stdout={listing.stdout!r}"
    )

    # The same folder the per-turn assembly reads parses cleanly.
    from horizon.tools.todos import read_todos

    todos = await read_todos()
    assert todos, "read_todos returned no items after the agent's turn"
    assert all(t.get("content") for t in todos)
