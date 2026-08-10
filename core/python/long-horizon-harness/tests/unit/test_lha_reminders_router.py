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

"""/lha/reminders — list + cancel, user-scoped, over the shared reminder store."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from horizon.api.reminders import attach_reminders_routes
from horizon.scheduler.store import (
    Reminder,
    get_reminder_store,
    reset_reminder_store,
)

pytestmark = pytest.mark.asyncio


def _reminder(
    rid: str,
    user_id: str,
    fire_at: datetime,
    *,
    message: str = "do the thing",
    recurrence: str | None = None,
) -> Reminder:
    return Reminder(
        id=rid,
        user_id=user_id,
        app_name="lha",
        channel="cli",
        recipient_id=user_id,
        message=message,
        fire_at=fire_at,
        recurrence=recurrence,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture()
def app_client(monkeypatch):
    monkeypatch.setenv("LHA_AUTH_MODE", "dev")
    monkeypatch.setenv("LHA_DEV_USER_ID", "alice@x")
    monkeypatch.delenv("LHA_REMINDER_STORE", raising=False)
    reset_reminder_store()
    app = FastAPI()
    attach_reminders_routes(app)
    yield TestClient(app)
    reset_reminder_store()


async def test_list_returns_user_reminders_sorted_by_fire_at(app_client):
    store = get_reminder_store()
    await store.add(
        _reminder("r2", "alice@x", datetime(2026, 6, 10, 9, tzinfo=UTC))
    )
    await store.add(
        _reminder("r1", "alice@x", datetime(2026, 6, 5, 9, tzinfo=UTC))
    )
    body = app_client.get("/lha/reminders").json()
    ids = [r["id"] for r in body["reminders"]]
    assert ids == ["r1", "r2"]
    assert body["reminders"][0]["fire_at"] == "2026-06-05T09:00:00+00:00"
    assert body["reminders"][0]["message"] == "do the thing"


async def test_list_excludes_other_users(app_client):
    store = get_reminder_store()
    await store.add(
        _reminder("mine", "alice@x", datetime(2026, 6, 5, tzinfo=UTC))
    )
    await store.add(
        _reminder("theirs", "bob@x", datetime(2026, 6, 5, tzinfo=UTC))
    )
    ids = [
        r["id"] for r in app_client.get("/lha/reminders").json()["reminders"]
    ]
    assert ids == ["mine"]


async def test_delete_removes_own_reminder(app_client):
    store = get_reminder_store()
    await store.add(
        _reminder("r1", "alice@x", datetime(2026, 6, 5, tzinfo=UTC))
    )
    r = app_client.delete("/lha/reminders/r1")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert app_client.get("/lha/reminders").json()["reminders"] == []


async def test_delete_other_users_reminder_is_404_and_survives(app_client):
    store = get_reminder_store()
    await store.add(
        _reminder("theirs", "bob@x", datetime(2026, 6, 5, tzinfo=UTC))
    )
    r = app_client.delete("/lha/reminders/theirs")
    assert r.status_code == 404
    assert await store.list_for_user("bob@x") != []


async def test_delete_unknown_id_is_404(app_client):
    r = app_client.delete("/lha/reminders/nope")
    assert r.status_code == 404


async def test_list_serializes_recurrence_and_channel(app_client):
    store = get_reminder_store()
    rem = _reminder(
        "r1", "alice@x", datetime(2026, 6, 5, 9, tzinfo=UTC), recurrence="daily"
    )
    from dataclasses import replace

    await store.add(replace(rem, channel="telegram"))
    item = app_client.get("/lha/reminders").json()["reminders"][0]
    assert item["recurrence"] == "daily"
    assert item["channel"] == "telegram"
