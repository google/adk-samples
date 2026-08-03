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

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from horizon.environment_context import set_active_environment
from tests.unit.test_todos_store import _FsEnv  # reuse the temp-dir env stub


@pytest.fixture
def env(tmp_path: Path) -> _FsEnv:
    e = _FsEnv(tmp_path)
    set_active_environment(e)
    return e


def _run(coro):
    return asyncio.run(coro)


# -----------------------------------------------------------------------------
# Surface
# -----------------------------------------------------------------------------


def test_write_todos_is_async_callable():
    from horizon.tools.todos import write_todos

    assert inspect.iscoroutinefunction(write_todos)


def test_no_session_state_param():
    from horizon.tools.todos import write_todos

    sig = inspect.signature(write_todos)
    assert "tool_context" in sig.parameters
    assert "state" not in sig.parameters


def _tool_context(session_id: str):
    return SimpleNamespace(
        _invocation_context=SimpleNamespace(
            session=SimpleNamespace(id=session_id)
        )
    )


def test_tool_context_scopes_to_session_folder(env, tmp_path):
    from horizon.tools.todos import read_todos, write_todos

    result = _run(
        write_todos(
            todos=[{"content": "scoped", "status": "pending"}],
            tool_context=_tool_context("s1"),
        )
    )
    assert result["success"] is True
    folder = tmp_path / "lha" / "todos" / "s1"
    assert sorted(p.name for p in folder.glob("*.md")) == ["001-scoped.md"]
    assert [t["content"] for t in _run(read_todos(session_id="s1"))] == [
        "scoped"
    ]


def test_no_tool_context_writes_flat_board(env, tmp_path):
    from horizon.tools.todos import write_todos

    _run(write_todos(todos=[{"content": "flat", "status": "pending"}]))
    assert (tmp_path / "lha" / "todos" / "001-flat.md").exists()


# -----------------------------------------------------------------------------
# Persistence contract (files, not state)
# -----------------------------------------------------------------------------


def test_writes_files_under_lha_todos(env, tmp_path):
    from horizon.tools.todos import write_todos

    result = _run(
        write_todos(
            todos=[
                {
                    "content": "step one",
                    "status": "in_progress",
                    "priority": "high",
                },
                {"content": "step two", "status": "pending"},
            ]
        )
    )
    assert result["success"] is True
    folder = tmp_path / "lha" / "todos"
    assert sorted(p.name for p in folder.glob("*.md")) == [
        "001-step-one.md",
        "002-step-two.md",
    ]


def test_overwrites_previous_list_wholesale(env, tmp_path):
    from horizon.tools.todos import write_todos

    _run(write_todos(todos=[{"content": "old", "status": "pending"}]))
    _run(write_todos(todos=[{"content": "new", "status": "pending"}]))
    folder = tmp_path / "lha" / "todos"
    assert sorted(p.name for p in folder.glob("*.md")) == ["001-new.md"]


def test_empty_list_clears_folder(env, tmp_path):
    from horizon.tools.todos import write_todos

    _run(write_todos(todos=[{"content": "old", "status": "pending"}]))
    result = _run(write_todos(todos=[]))
    folder = tmp_path / "lha" / "todos"
    assert result["success"] is True
    assert sorted(folder.glob("*.md")) == []


def test_returns_rendered_view(env):
    from horizon.tools.todos import write_todos

    result = _run(
        write_todos(todos=[{"content": "do it", "status": "pending"}])
    )
    assert "# Todos" in result["rendered"]
    assert "do it" in result["rendered"]


# -----------------------------------------------------------------------------
# Validation — list shape
# -----------------------------------------------------------------------------


def test_non_list_rejected(env):
    from horizon.tools.todos import write_todos

    result = _run(write_todos(todos="nope"))  # type: ignore[arg-type]
    assert result["success"] is False
    assert "list" in result["error"].lower()


def test_item_not_a_dict_rejected(env):
    from horizon.tools.todos import write_todos

    result = _run(write_todos(todos=["x"]))  # type: ignore[list-item]
    assert result["success"] is False


def test_missing_content_rejected(env):
    from horizon.tools.todos import write_todos

    result = _run(write_todos(todos=[{"status": "pending"}]))
    assert result["success"] is False
    assert "content" in result["error"].lower()


def test_blank_content_rejected(env):
    from horizon.tools.todos import write_todos

    result = _run(write_todos(todos=[{"content": "  ", "status": "pending"}]))
    assert result["success"] is False


# -----------------------------------------------------------------------------
# Validation — enums
# -----------------------------------------------------------------------------


def test_invalid_status_rejected(env):
    from horizon.tools.todos import write_todos

    result = _run(write_todos(todos=[{"content": "x", "status": "done"}]))
    assert result["success"] is False
    assert "status" in result["error"].lower()


def test_missing_status_defaults_pending(env, tmp_path):
    from horizon.tools.todos import write_todos

    _run(write_todos(todos=[{"content": "x"}]))
    text = (tmp_path / "lha" / "todos" / "001-x.md").read_text()
    assert "status: pending" in text


def test_invalid_priority_rejected(env):
    from horizon.tools.todos import write_todos

    result = _run(
        write_todos(
            todos=[{"content": "x", "status": "pending", "priority": "urgent"}]
        )
    )
    assert result["success"] is False
    assert "priority" in result["error"].lower()


# -----------------------------------------------------------------------------
# One-in_progress rule: WARN, never reject
# -----------------------------------------------------------------------------


def test_multiple_in_progress_persists_with_warning(env, tmp_path):
    from horizon.tools.todos import write_todos

    result = _run(
        write_todos(
            todos=[
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "in_progress"},
            ]
        )
    )
    assert result["success"] is True
    folder = tmp_path / "lha" / "todos"
    assert len(list(folder.glob("*.md"))) == 2
    assert result.get("warning")
    assert "in_progress" in result["warning"]


def test_single_in_progress_no_warning(env):
    from horizon.tools.todos import write_todos

    result = _run(
        write_todos(todos=[{"content": "a", "status": "in_progress"}])
    )
    assert result.get("warning") is None


def test_write_todos_registered_on_root_agent():
    from horizon.agent import root_agent
    from horizon.tools.todos import write_todos

    assert write_todos in root_agent.tools
