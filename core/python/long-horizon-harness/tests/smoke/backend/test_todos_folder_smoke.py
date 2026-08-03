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

"""``write_todos`` folder write + per-turn render through the env interface.

Direct tool-call rather than LLM-driven — same pattern as
``test_terminal_tool.py`` / ``test_file_ops_tool.py``. The smoke contract
is "the file-based todo store and the per-turn folder walker work against
the real (sandbox) environment"; LLM behavior is covered by ``tests/eval``.

This is the test that proves the embedded-Python folder walker
(``read_todos`` / ``render_todos``) and the reconcile write path run
correctly in-sandbox, not just against the local _FsEnv stub.
"""

from __future__ import annotations

import pytest
from google.adk.environment import BaseEnvironment

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _read_todo_file(env: BaseEnvironment, name: str) -> str:
    raw = await env.read_file(
        env.working_dir.resolve() / "lha" / "todos" / name
    )
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


async def test_write_todos_creates_files_with_frontmatter(
    active_env: BaseEnvironment,
) -> None:
    from horizon.tools.todos import write_todos

    result = await write_todos(
        todos=[
            {
                "content": "smoke step one",
                "status": "in_progress",
                "priority": "high",
            },
            {"content": "smoke step two", "status": "pending"},
        ]
    )
    assert result["success"] is True

    # Files exist under <workspace>/lha/todos/, one per todo, read back via the interface.
    one = await _read_todo_file(active_env, "001-smoke-step-one.md")
    two = await _read_todo_file(active_env, "002-smoke-step-two.md")

    assert one.startswith("---\n")
    assert "status: in_progress" in one
    assert "priority: high" in one
    assert one.rstrip().endswith("smoke step one")

    assert "status: pending" in two
    assert "priority:" not in two
    assert two.rstrip().endswith("smoke step two")


async def test_per_turn_walker_renders_block_from_sandbox(
    active_env: BaseEnvironment,
) -> None:
    from horizon.tools.todos import read_todos, render_todos, write_todos

    await write_todos(
        todos=[
            {"content": "walk a", "status": "completed"},
            {"content": "walk b", "status": "in_progress"},
            {"content": "walk c", "status": "pending"},
        ]
    )

    # The per-turn read path: env.execute folder walker → parse → render.
    todos = await read_todos()
    assert [t["content"] for t in todos] == ["walk a", "walk b", "walk c"]

    rendered = render_todos(todos)
    assert rendered.splitlines()[0] == "# Todos"
    assert "[x] walk a" in rendered
    assert "[~] walk b" in rendered
    assert "[ ] walk c" in rendered


async def test_reconcile_renames_and_removes_without_orphans(
    active_env: BaseEnvironment,
) -> None:
    from horizon.tools.todos import read_todos, write_todos

    await write_todos(
        todos=[
            {"content": "keeper", "status": "pending"},
            {"content": "doomed item", "status": "pending"},
            {"content": "rename me", "status": "pending"},
        ]
    )

    # Drop one item and rename another (new slug → new filename). The folder
    # must match the new list exactly: no orphan files left behind.
    await write_todos(
        todos=[
            {"content": "keeper", "status": "completed"},
            {"content": "renamed already", "status": "in_progress"},
        ]
    )

    todos = await read_todos()
    assert [t["content"] for t in todos] == ["keeper", "renamed already"]

    listing = await active_env.execute(
        f"ls {active_env.working_dir.resolve() / 'lha' / 'todos'}"
    )
    names = sorted(n for n in listing.stdout.split() if n.endswith(".md"))
    assert names == ["001-keeper.md", "002-renamed-already.md"]
