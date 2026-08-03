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

"""CUJ-12: reminders survive a Cloud Run revision restart.

The whole point of PR2's Postgres backend is that an in-flight reminder
isn't lost when the instance serving the tick endpoint gets recycled.
Two scenarios cover the surface:

1. **Revision restart**: open a ``PostgresReminderStore``, write a future
   reminder, ``close()`` the pool (simulating Cloud Run draining the
   instance), open a *fresh* store on the same DSN, run a tick — the
   reminder fires and is gone afterward. Proves the data lives in
   Postgres, not in process memory.

2. **FOR UPDATE SKIP LOCKED race**: two distinct stores (separate
   asyncpg pools, simulating two Cloud Run replicas) call ``claim_due``
   concurrently on the same row set. The union of claimed reminders has
   no duplicates, the total claimed equals the number of due rows. If
   SKIP LOCKED were missing, one of the two would either block or
   double-claim.

Gated on ``LHA_TEST_POSTGRES_URL`` like the unit-test live round-trip
suite — skipped on machines without a local Postgres so CI stays green.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

pytestmark = pytest.mark.asyncio

_DSN_ENV = "LHA_TEST_POSTGRES_URL"
_skip_no_db = pytest.mark.skipif(
    not os.environ.get(_DSN_ENV),
    reason=f"set {_DSN_ENV} to a Postgres DSN to enable CUJ-12 durability probes",
)


def _build_app() -> FastAPI:
    from horizon.scheduler import tick_endpoint

    app = FastAPI()
    app.include_router(tick_endpoint.router)
    return app


def _reminder(
    *,
    user_id: str = "u-durability",
    fire_at: datetime,
    message: str = "ping",
    recurrence: str | None = None,
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


@pytest_asyncio.fixture
async def clean_table():
    """Drop the reminders table between tests so rows from one probe
    don't leak into another. Yields the DSN."""
    import asyncpg

    dsn = os.environ[_DSN_ENV]
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP TABLE IF EXISTS reminders")
    finally:
        await conn.close()
    yield dsn
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP TABLE IF EXISTS reminders")
    finally:
        await conn.close()


@_skip_no_db
async def test_reminder_survives_pool_restart(monkeypatch, clean_table):
    """Write → close pool → reopen pool → tick → reminder fires.

    Models a Cloud Run revision being recycled between the write that
    scheduled the reminder and the tick that fires it.
    """
    from horizon.scheduler import store as store_mod
    from horizon.scheduler import tick_endpoint
    from horizon.scheduler.postgres_store import PostgresReminderStore

    dsn = clean_table

    monkeypatch.setenv("LHA_SCHEDULER_AUTH_DISABLED", "1")

    # Revision A: write a due reminder, then drain.
    revision_a = PostgresReminderStore(dsn=dsn)
    past = datetime.now(UTC) - timedelta(minutes=5)
    rid = await revision_a.add(
        _reminder(fire_at=past, message="survives restart")
    )
    assert rid
    await revision_a.close()

    # Revision B: fresh pool, fresh process, same DSN. The router uses
    # ``get_reminder_store`` so wire B in as the singleton.
    revision_b = PostgresReminderStore(dsn=dsn)
    store_mod.reset_reminder_store()
    monkeypatch.setattr(store_mod, "_singleton", revision_b)
    assert store_mod.get_reminder_store() is revision_b

    pushes: list[dict[str, Any]] = []

    async def fake_push(**kwargs):
        pushes.append(kwargs)
        return {"delivered": True}

    monkeypatch.setattr(tick_endpoint, "push_to_user", fake_push)

    try:
        transport = httpx.ASGITransport(app=_build_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as ac:
            resp = await ac.post("/scheduler/tick")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"fired": 1}
        assert len(pushes) == 1
        assert pushes[0]["text"] == "survives restart"

        remaining = await revision_b.list_for_user("u-durability")
        assert remaining == [], (
            "one-shot must be gone after firing on new revision"
        )
    finally:
        await revision_b.close()
        store_mod.reset_reminder_store()


@_skip_no_db
async def test_skip_locked_prevents_double_claim_across_pools(clean_table):
    """Two replicas, two pools, one ``claim_due`` per pool, run concurrently.

    The union of claimed IDs is the set of due reminders; no row appears
    in both lists. If ``FOR UPDATE SKIP LOCKED`` were missing, the two
    transactions would either deadlock or one would re-read rows that
    the other had already advanced.
    """
    from horizon.scheduler.postgres_store import PostgresReminderStore

    dsn = clean_table

    seeder = PostgresReminderStore(dsn=dsn)
    past = datetime.now(UTC) - timedelta(minutes=5)
    seeded_ids = []
    for i in range(40):
        rid = await seeder.add(
            _reminder(fire_at=past, message=f"row-{i}", user_id="u-race")
        )
        seeded_ids.append(rid)
    await seeder.close()

    replica_one = PostgresReminderStore(dsn=dsn)
    replica_two = PostgresReminderStore(dsn=dsn)

    try:
        cursor = datetime.now(UTC)
        claimed_one, claimed_two = await asyncio.gather(
            replica_one.claim_due(cursor),
            replica_two.claim_due(cursor),
        )
    finally:
        await replica_one.close()
        await replica_two.close()

    ids_one = {r.id for r in claimed_one}
    ids_two = {r.id for r in claimed_two}

    assert ids_one.isdisjoint(ids_two), (
        f"SKIP LOCKED failed: {len(ids_one & ids_two)} rows were claimed by "
        f"both replicas — overlap={sorted(ids_one & ids_two)[:5]}"
    )
    assert ids_one | ids_two == set(seeded_ids), (
        "the union of two concurrent claims must equal the due set; "
        f"missing={set(seeded_ids) - (ids_one | ids_two)}"
    )

    # Sanity: the table is empty afterward (all one-shots deleted).
    audit = PostgresReminderStore(dsn=dsn)
    try:
        remaining = await audit.list_for_user("u-race")
        assert remaining == [], (
            f"all one-shots should be gone; {len(remaining)} survived"
        )
    finally:
        await audit.close()


@_skip_no_db
async def test_recurring_reminder_persists_advanced_fire_at(
    monkeypatch, clean_table
):
    """Daily reminder fires, ``fire_at`` advances by 24h, the row survives
    a pool restart with the advanced timestamp intact.

    This catches a class of bug where recurrence is computed in memory
    but the UPDATE never reaches Postgres — the new revision would
    re-fire the same reminder over and over.
    """
    from horizon.scheduler import store as store_mod
    from horizon.scheduler import tick_endpoint
    from horizon.scheduler.postgres_store import PostgresReminderStore

    dsn = clean_table

    monkeypatch.setenv("LHA_SCHEDULER_AUTH_DISABLED", "1")

    revision_a = PostgresReminderStore(dsn=dsn)
    original_fire_at = datetime.now(UTC) - timedelta(minutes=5)
    await revision_a.add(
        _reminder(
            fire_at=original_fire_at,
            recurrence="daily",
            message="daily survivor",
            user_id="u-recurring",
        )
    )

    store_mod.reset_reminder_store()
    monkeypatch.setattr(store_mod, "_singleton", revision_a)

    async def fake_push(**kwargs):
        return {"delivered": True}

    monkeypatch.setattr(tick_endpoint, "push_to_user", fake_push)

    try:
        transport = httpx.ASGITransport(app=_build_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as ac:
            resp = await ac.post("/scheduler/tick")
        assert resp.status_code == 200
        assert resp.json() == {"fired": 1}
    finally:
        await revision_a.close()
        store_mod.reset_reminder_store()

    # New revision: read what got persisted. The advance must be on disk,
    # not just in revision A's process memory.
    revision_b = PostgresReminderStore(dsn=dsn)
    try:
        remaining = await revision_b.list_for_user("u-recurring")
        assert len(remaining) == 1, "recurring reminder must survive its fire"
        assert remaining[0].fire_at == original_fire_at + timedelta(days=1)
        assert remaining[0].recurrence == "daily"
    finally:
        await revision_b.close()
