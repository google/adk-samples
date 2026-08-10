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

"""Tick fires a reminder into a tagged, persisted session and pushes best-effort."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.adk.sessions import InMemorySessionService

from horizon.infrastructure.constants import (
    APP_NAME,
    SCHEDULER_JOB_KEY,
    SESSION_SOURCE_KEY,
)

pytestmark = pytest.mark.asyncio


@dataclass
class _StubRunner:
    session_service: InMemorySessionService = field(
        default_factory=InMemorySessionService
    )


def _reminder(message: str = "summarize my unread mail"):
    from horizon.scheduler.store import Reminder

    return Reminder(
        id="",
        user_id="alice",
        app_name=APP_NAME,
        channel="cli",
        recipient_id="alice",
        message=message,
        fire_at=datetime(2026, 1, 1, 0, tzinfo=UTC),
        recurrence=None,
        created_at=datetime(2026, 1, 1, 0, tzinfo=UTC),
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    from horizon.scheduler import store as store_mod

    store_mod.reset_reminder_store()
    monkeypatch.delenv("LHA_REMINDER_STORE", raising=False)
    monkeypatch.setenv("LHA_SCHEDULER_AUTH_DISABLED", "1")
    yield
    store_mod.reset_reminder_store()


def _app_with_runner(runner: _StubRunner) -> FastAPI:
    from horizon.scheduler import tick_endpoint

    app = FastAPI()
    app.include_router(tick_endpoint.router)
    app.state.runner = runner
    # Tests that stub _run_reminder_turn just need a non-None handler; tests
    # exercising the real turn override this with a fake handler.
    app.state.a2a_handler = object()
    return app


async def test_firing_creates_tagged_session(monkeypatch):
    from horizon.scheduler import tick_endpoint
    from horizon.scheduler.store import get_reminder_store

    async def fake_turn(handler, *, user_id, session_id, message):
        return "Here is your mail summary."

    pushes: list[dict[str, Any]] = []

    async def fake_push(**kwargs):
        pushes.append(kwargs)
        return {"delivered": True}

    monkeypatch.setattr(tick_endpoint, "_run_reminder_turn", fake_turn)
    monkeypatch.setattr(tick_endpoint, "push_to_user", fake_push)

    runner = _StubRunner()
    await get_reminder_store().add(_reminder())

    client = TestClient(_app_with_runner(runner))
    resp = client.post("/scheduler/tick")
    assert resp.json()["fired"] == 1

    listed = await runner.session_service.list_sessions(
        app_name=APP_NAME, user_id="alice"
    )
    assert len(listed.sessions) == 1
    full = await runner.session_service.get_session(
        app_name=APP_NAME, user_id="alice", session_id=listed.sessions[0].id
    )
    assert full.state[SESSION_SOURCE_KEY] == "scheduler"
    assert full.state[SCHEDULER_JOB_KEY] == "reminder"
    # Decoupled delivery pushes the agent's result, not the raw reminder text.
    assert pushes[0]["text"] == "Here is your mail summary."


async def test_firing_drives_a2a_handler_with_context_and_identity(monkeypatch):
    """The reminder turn runs through the shared A2A handler, targeting the
    pre-tagged session by context_id and under the reminder's user_id — so the
    executor records a Task the web UI can render, exactly like a normal chat.
    """
    from google.adk.a2a import _compat as a2a_compat

    from horizon.auth import get_user_id_from_context
    from horizon.scheduler import tick_endpoint
    from horizon.scheduler.store import get_reminder_store

    captured: dict[str, Any] = {}

    class FakeHandler:
        async def on_message_send(self, params, context=None):
            captured["context_id"] = params.message.context_id
            captured["user_id"] = get_user_id_from_context()
            captured["text_in"] = a2a_compat.part_text(params.message.parts[0])
            reply = a2a_compat.make_message(
                message_id="reply-1",
                role=a2a_compat.ROLE_AGENT,
                parts=[a2a_compat.make_text_part("Here is your mail summary.")],
                context_id=params.message.context_id,
            )
            return a2a_compat.make_task(
                id="task-1",
                context_id=params.message.context_id,
                status=a2a_compat.make_task_status(
                    a2a_compat.TS_COMPLETED, message=reply
                ),
                history=[params.message, reply],
            )

    pushes: list[dict[str, Any]] = []

    async def fake_push(**kwargs):
        pushes.append(kwargs)
        return {"delivered": True}

    monkeypatch.setattr(tick_endpoint, "push_to_user", fake_push)

    runner = _StubRunner()
    await get_reminder_store().add(_reminder())

    app = _app_with_runner(runner)
    app.state.a2a_handler = FakeHandler()
    client = TestClient(app)
    resp = client.post("/scheduler/tick")
    assert resp.json()["fired"] == 1

    listed = await runner.session_service.list_sessions(
        app_name=APP_NAME, user_id="alice"
    )
    assert len(listed.sessions) == 1
    session_id = listed.sessions[0].id

    # Drove the shared handler against the pre-tagged session, as the reminder's user.
    assert captured["context_id"] == session_id
    assert captured["user_id"] == "alice"
    assert captured["text_in"] == "summarize my unread mail"
    # Identity scope is reset once the turn completes.
    assert get_user_id_from_context() is None
    # Delivery pushes the agent's final text, extracted from the returned Task.
    assert pushes[0]["text"] == "Here is your mail summary."


async def test_firing_tolerates_absent_gateway(monkeypatch):
    from horizon.scheduler import tick_endpoint
    from horizon.scheduler.store import get_reminder_store

    async def fake_turn(handler, *, user_id, session_id, message):
        return ""

    async def boom_push(**kwargs):
        raise RuntimeError("no gateway configured")

    monkeypatch.setattr(tick_endpoint, "_run_reminder_turn", fake_turn)
    monkeypatch.setattr(tick_endpoint, "push_to_user", boom_push)

    runner = _StubRunner()
    await get_reminder_store().add(_reminder())

    client = TestClient(_app_with_runner(runner))
    resp = client.post("/scheduler/tick")
    # Push raised, but the session was still created and the tick succeeded.
    assert resp.status_code == 200
    assert resp.json()["fired"] == 1
    listed = await runner.session_service.list_sessions(
        app_name=APP_NAME, user_id="alice"
    )
    assert len(listed.sessions) == 1
