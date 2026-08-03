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

"""CUJ-10: a scheduled reminder fires exactly once, even under overlapping ticks.

Mounts the real tick router against an in-memory store, swaps ``push_to_user``
for a recorder, and:

1. Single reminder → one tick → push delivered with the right payload, store
   empty afterward (the one-shot was removed).
2. N reminders → M concurrent ticks via ``httpx.AsyncClient`` → total fires
   across all tick responses == N. This is the live exercise of the
   ``InMemoryReminderStore.claim_due`` race fix; if two ticks both saw the
   same row before either claimed it, push would be called more than N times.
3. Recurring (``daily``) reminder → one tick → advanced ``fire_at`` is 24h
   later, push delivered once. Proves recurrence is in claim_due, not on
   the deleted ``mark_fired`` path.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.asyncio


def _build_app() -> FastAPI:
    from horizon.scheduler import tick_endpoint

    app = FastAPI()
    app.include_router(tick_endpoint.router)
    return app


def _reminder(
    *,
    user_id: str = "u1",
    fire_at: datetime,
    recurrence: str | None = None,
    message: str = "ping",
):
    from horizon.scheduler.store import Reminder

    return Reminder(
        id="",
        user_id=user_id,
        app_name="lha",
        channel="cli",
        recipient_id=user_id,
        message=message,
        fire_at=fire_at,
        recurrence=recurrence,
        created_at=datetime.now(UTC),
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    from horizon.scheduler import store as store_mod

    store_mod.reset_reminder_store()
    monkeypatch.setenv("LHA_SCHEDULER_AUTH_DISABLED", "1")
    monkeypatch.delenv("LHA_REMINDER_STORE", raising=False)
    yield
    store_mod.reset_reminder_store()


async def test_single_reminder_fires_once_and_is_removed(monkeypatch):
    from horizon.scheduler import tick_endpoint
    from horizon.scheduler.store import get_reminder_store

    pushes: list[dict[str, Any]] = []

    async def fake_push(**kwargs):
        pushes.append(kwargs)
        return {"delivered": True}

    monkeypatch.setattr(tick_endpoint, "push_to_user", fake_push)

    past = datetime.now(UTC) - timedelta(minutes=5)
    store = get_reminder_store()
    rid = await store.add(_reminder(fire_at=past, message="say hi"))

    transport = httpx.ASGITransport(app=_build_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t"
    ) as ac:
        resp = await ac.post("/scheduler/tick")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"fired": 1}
    assert len(pushes) == 1
    assert pushes[0]["text"] == "say hi"
    assert pushes[0]["user_id"] == "u1"

    remaining = await store.list_for_user("u1")
    assert remaining == [], f"one-shot {rid} should be gone after fire"


async def test_concurrent_ticks_do_not_double_fire(monkeypatch):
    from horizon.scheduler import tick_endpoint
    from horizon.scheduler.store import get_reminder_store

    pushes: list[dict[str, Any]] = []
    push_lock = asyncio.Lock()

    async def fake_push(**kwargs):
        async with push_lock:
            pushes.append(kwargs)
        await asyncio.sleep(0.01)
        return {"delivered": True}

    monkeypatch.setattr(tick_endpoint, "push_to_user", fake_push)

    past = datetime.now(UTC) - timedelta(minutes=5)
    store = get_reminder_store()
    for i in range(20):
        await store.add(_reminder(fire_at=past, message=f"msg-{i}"))

    transport = httpx.ASGITransport(app=_build_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t"
    ) as ac:
        responses = await asyncio.gather(
            *(ac.post("/scheduler/tick") for _ in range(5))
        )

    assert all(r.status_code == 200 for r in responses)
    total_fired = sum(r.json().get("fired", 0) for r in responses)

    assert total_fired == 20, (
        f"expected each reminder fired exactly once across 5 concurrent ticks; "
        f"got {total_fired} total fires, {len(pushes)} push calls"
    )
    assert len(pushes) == 20

    delivered_messages = sorted(p["text"] for p in pushes)
    assert delivered_messages == sorted(f"msg-{i}" for i in range(20))

    remaining = await store.list_for_user("u1")
    assert remaining == []


async def test_recurring_reminder_advances_fire_at(monkeypatch):
    from horizon.scheduler import tick_endpoint
    from horizon.scheduler.store import get_reminder_store

    pushes: list[dict[str, Any]] = []

    async def fake_push(**kwargs):
        pushes.append(kwargs)
        return {"delivered": True}

    monkeypatch.setattr(tick_endpoint, "push_to_user", fake_push)

    original_fire_at = datetime.now(UTC) - timedelta(minutes=5)
    store = get_reminder_store()
    await store.add(
        _reminder(
            fire_at=original_fire_at, recurrence="daily", message="daily ping"
        )
    )

    transport = httpx.ASGITransport(app=_build_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t"
    ) as ac:
        resp = await ac.post("/scheduler/tick")

    assert resp.status_code == 200
    assert resp.json() == {"fired": 1}
    assert len(pushes) == 1

    remaining = await store.list_for_user("u1")
    assert len(remaining) == 1, "recurring reminder must survive its fire"
    advanced = remaining[0]
    assert advanced.fire_at == original_fire_at + timedelta(days=1)
    assert advanced.recurrence == "daily"
