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
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from google.adk.models import LlmRequest
from google.genai import types

from horizon.environment_context import set_active_environment
from tests.unit.test_todos_store import _FsEnv


@pytest.fixture
def env(tmp_path: Path) -> _FsEnv:
    e = _FsEnv(tmp_path)
    set_active_environment(e)
    return e


def _request():
    captured: list[str] = []

    class _Cfg:
        system_instruction = "base"

    class _Req:
        config = _Cfg()
        model = "gemini-3.6-flash"
        tools_dict: ClassVar[dict] = {}

        def append_instructions(self, blocks):
            captured.extend(blocks)

    return _Req(), captured


def _ctx(tmp_path: Path, state: dict | None = None):
    return SimpleNamespace(state=state or {}, cwd=str(tmp_path))


def _reminder_request() -> LlmRequest:
    return LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])]
    )


def _reminder_texts(req: LlmRequest) -> str:
    out: list[str] = []
    for content in req.contents:
        for part in content.parts or []:
            if getattr(part, "text", None) and "<system-reminder>" in part.text:
                out.append(part.text)
    return "\n".join(out)


def test_todos_ride_reminder_tail_not_instruction(env, tmp_path):
    from horizon.conversation.reminders import reminder_injection_callback
    from horizon.conversation.system_prompt import (
        system_prompt_assembly_callback,
    )
    from horizon.tools.todos import write_todos

    asyncio.run(
        write_todos(todos=[{"content": "ship it", "status": "in_progress"}])
    )

    # Todos must NOT land in the cached instruction tier.
    req, captured = _request()
    asyncio.run(system_prompt_assembly_callback(_ctx(tmp_path), req))
    joined = "\n".join(captured)
    assert "# Todos" not in joined
    assert "ship it" not in joined

    # Todos DO render into the trailing <system-reminder> tail.
    rem_req = _reminder_request()
    asyncio.run(reminder_injection_callback(_ctx(tmp_path), rem_req))
    tail = _reminder_texts(rem_req)
    assert "# Todos" in tail
    assert "ship it" in tail


def test_prompt_omits_todos_when_folder_empty(env, tmp_path):
    from horizon.conversation.system_prompt import (
        system_prompt_assembly_callback,
    )

    req, captured = _request()
    asyncio.run(system_prompt_assembly_callback(_ctx(tmp_path), req))

    assert "# Todos" not in "\n".join(captured)


def _session_ctx(tmp_path: Path, session_id: str):
    return SimpleNamespace(
        state={},
        cwd=str(tmp_path),
        _invocation_context=SimpleNamespace(
            session=SimpleNamespace(id=session_id)
        ),
    )


def test_build_todos_reminder_renders_only_current_session(env):
    from horizon.conversation.reminders import build_todos_reminder
    from horizon.tools.todos import reconcile_todos

    asyncio.run(
        reconcile_todos(
            [{"id": "s1", "content": "s1 task", "status": "pending"}],
            session_id="s1",
        )
    )

    rendered = asyncio.run(build_todos_reminder(session_id="s1"))
    assert rendered is not None and "s1 task" in rendered
    assert asyncio.run(build_todos_reminder(session_id="s2")) is None


def test_reminder_callback_injects_current_session_board(env, tmp_path):
    from horizon.conversation.reminders import reminder_injection_callback
    from horizon.tools.todos import reconcile_todos

    asyncio.run(
        reconcile_todos(
            [{"id": "s1", "content": "s1 only", "status": "in_progress"}],
            session_id="s1",
        )
    )
    asyncio.run(
        reconcile_todos(
            [{"id": "s2", "content": "s2 only", "status": "pending"}],
            session_id="s2",
        )
    )

    rem_req = _reminder_request()
    asyncio.run(
        reminder_injection_callback(_session_ctx(tmp_path, "s1"), rem_req)
    )
    tail = _reminder_texts(rem_req)
    assert "s1 only" in tail
    assert "s2 only" not in tail
