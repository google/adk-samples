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

"""``InMemoryReminderStore`` — async dict-backed reminder persistence.

Covers the surface every ``ReminderStore`` impl must honor (Postgres impl
in PR2 will share these tests via the same protocol):

* round-trip ``add`` → ``list_for_user``
* ``claim_due`` atomically returns and advances/removes due reminders
* ``cancel`` is user-scoped — user A cannot cancel user B's reminder
* the ``get_reminder_store()`` factory honors ``LHA_REMINDER_STORE``
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

pytestmark = pytest.mark.asyncio


def _at(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _reminder(
    user_id: str = "u1",
    *,
    fire_at: datetime | None = None,
    message: str = "remember the milk",
    recurrence: str | None = None,
    channel: str = "cli",
    recipient_id: str = "u1",
    app_name: str = "lha",
):
    from horizon.scheduler.store import Reminder

    return Reminder(
        id="",
        user_id=user_id,
        app_name=app_name,
        channel=channel,
        recipient_id=recipient_id,
        message=message,
        fire_at=fire_at or _at(2026, 5, 20, 9),
        recurrence=recurrence,
        created_at=_at(2026, 5, 19, 12),
    )


# =============================================================================
# Module surface
# =============================================================================


async def test_module_exposes_protocol_and_impl():
    from horizon.scheduler import store

    assert hasattr(store, "Reminder")
    assert hasattr(store, "ReminderStore")
    assert hasattr(store, "InMemoryReminderStore")
    assert hasattr(store, "get_reminder_store")


# =============================================================================
# add / list_for_user
# =============================================================================


class TestAddAndList:
    async def test_add_assigns_id_and_persists(self):
        from horizon.scheduler.store import InMemoryReminderStore

        store = InMemoryReminderStore()
        rid = await store.add(_reminder())
        assert isinstance(rid, str) and rid
        listed = await store.list_for_user("u1")
        assert len(listed) == 1
        assert listed[0].id == rid
        assert listed[0].message == "remember the milk"

    async def test_list_for_user_scopes_by_user_id(self):
        from horizon.scheduler.store import InMemoryReminderStore

        store = InMemoryReminderStore()
        await store.add(_reminder(user_id="alice", message="alice's task"))
        await store.add(_reminder(user_id="bob", message="bob's task"))
        alice_items = await store.list_for_user("alice")
        bob_items = await store.list_for_user("bob")
        assert [r.message for r in alice_items] == ["alice's task"]
        assert [r.message for r in bob_items] == ["bob's task"]


# =============================================================================
# cancel — user-scoped
# =============================================================================


class TestCancel:
    async def test_owner_can_cancel(self):
        from horizon.scheduler.store import InMemoryReminderStore

        store = InMemoryReminderStore()
        rid = await store.add(_reminder(user_id="alice"))
        ok = await store.cancel(rid, "alice")
        assert ok is True
        assert await store.list_for_user("alice") == []

    async def test_other_user_cannot_cancel(self):
        from horizon.scheduler.store import InMemoryReminderStore

        store = InMemoryReminderStore()
        rid = await store.add(_reminder(user_id="alice"))
        ok = await store.cancel(rid, "bob")
        assert ok is False
        assert len(await store.list_for_user("alice")) == 1

    async def test_unknown_id_returns_false(self):
        from horizon.scheduler.store import InMemoryReminderStore

        store = InMemoryReminderStore()
        assert await store.cancel("nope", "alice") is False


# =============================================================================
# claim_due — atomic claim+advance to prevent duplicate firing
# =============================================================================


class TestClaimDue:
    async def test_claim_returns_due_rows_and_removes_one_shot(self):
        from horizon.scheduler.store import InMemoryReminderStore

        store = InMemoryReminderStore()
        await store.add(_reminder(fire_at=_at(2026, 5, 20, 8)))
        await store.add(_reminder(fire_at=_at(2026, 5, 20, 10)))  # not due
        claimed = await store.claim_due(_at(2026, 5, 20, 9))
        assert len(claimed) == 1
        assert claimed[0].fire_at == _at(2026, 5, 20, 8)
        # one-shot was removed; second claim returns nothing for it
        again = await store.claim_due(_at(2026, 5, 20, 9))
        assert again == []
        # the not-yet-due one is still present
        remaining = await store.list_for_user("u1")
        assert [r.fire_at for r in remaining] == [_at(2026, 5, 20, 10)]

    async def test_claim_advances_recurring_rows(self):
        from horizon.scheduler.store import InMemoryReminderStore

        store = InMemoryReminderStore()
        original = _at(2026, 5, 20, 9)
        await store.add(_reminder(fire_at=original, recurrence="daily"))
        claimed = await store.claim_due(original)
        assert len(claimed) == 1
        assert claimed[0].fire_at == original
        remaining = await store.list_for_user("u1")
        assert len(remaining) == 1
        assert remaining[0].fire_at == original + timedelta(days=1)

    async def test_claim_skips_and_removes_invalid_recurrence(self):
        from dataclasses import replace

        from horizon.scheduler.store import InMemoryReminderStore

        store = InMemoryReminderStore()
        rid_bad = await store.add(_reminder(fire_at=_at(2026, 5, 20, 7)))
        rid_good = await store.add(_reminder(fire_at=_at(2026, 5, 20, 8)))
        # Inject an invalid recurrence value bypassing the dataclass type hint
        bad = store._rows[rid_bad]
        store._rows[rid_bad] = replace(bad, recurrence="monthly")  # type: ignore[arg-type]
        claimed = await store.claim_due(_at(2026, 5, 20, 9))
        # The good row is still claimed; the bad row is dropped so it does not
        # keep crashing every tick.
        claimed_ids = {r.id for r in claimed}
        assert rid_good in claimed_ids
        assert await store.list_for_user("u1") == []

    async def test_claim_empty_when_nothing_due(self):
        from horizon.scheduler.store import InMemoryReminderStore

        store = InMemoryReminderStore()
        await store.add(_reminder(fire_at=_at(2026, 5, 20, 10)))
        assert await store.claim_due(_at(2026, 5, 20, 9)) == []


# =============================================================================
# Factory
# =============================================================================


class TestFactory:
    async def test_default_returns_in_memory(self, monkeypatch):
        from horizon.scheduler.store import (
            InMemoryReminderStore,
            get_reminder_store,
        )

        monkeypatch.delenv("LHA_REMINDER_STORE", raising=False)
        store = get_reminder_store()
        assert isinstance(store, InMemoryReminderStore)

    async def test_memory_env_returns_in_memory(self, monkeypatch):
        from horizon.scheduler.store import (
            InMemoryReminderStore,
            get_reminder_store,
        )

        monkeypatch.setenv("LHA_REMINDER_STORE", "memory")
        store = get_reminder_store()
        assert isinstance(store, InMemoryReminderStore)

    async def test_factory_is_singleton(self, monkeypatch):
        from horizon.scheduler.store import get_reminder_store

        monkeypatch.delenv("LHA_REMINDER_STORE", raising=False)
        a = get_reminder_store()
        b = get_reminder_store()
        assert a is b

    async def test_unknown_backend_raises(self, monkeypatch):
        from horizon.scheduler.store import (
            get_reminder_store,
            reset_reminder_store,
        )

        monkeypatch.setenv("LHA_REMINDER_STORE", "bogus")
        reset_reminder_store()
        with pytest.raises(ValueError):
            get_reminder_store()


# =============================================================================
# Recurrence enum validation
# =============================================================================


class TestRecurrence:
    async def test_compute_next_daily(self):
        from horizon.scheduler.store import compute_next_fire_at

        nxt = compute_next_fire_at(_at(2026, 5, 20, 9), "daily")
        assert nxt == _at(2026, 5, 20, 9) + timedelta(days=1)

    async def test_compute_next_weekly(self):
        from horizon.scheduler.store import compute_next_fire_at

        nxt = compute_next_fire_at(_at(2026, 5, 20, 9), "weekly")
        assert nxt == _at(2026, 5, 20, 9) + timedelta(days=7)

    async def test_compute_next_hourly(self):
        from horizon.scheduler.store import compute_next_fire_at

        nxt = compute_next_fire_at(_at(2026, 5, 20, 9), "hourly")
        assert nxt == _at(2026, 5, 20, 9) + timedelta(hours=1)

    async def test_compute_next_none_returns_none(self):
        from horizon.scheduler.store import compute_next_fire_at

        assert compute_next_fire_at(_at(2026, 5, 20, 9), None) is None

    async def test_compute_next_unknown_raises(self):
        from horizon.scheduler.store import compute_next_fire_at

        with pytest.raises(ValueError):
            compute_next_fire_at(_at(2026, 5, 20, 9), "yearly")
