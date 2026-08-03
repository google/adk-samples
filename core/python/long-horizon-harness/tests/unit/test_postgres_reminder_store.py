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

"""``PostgresReminderStore`` — asyncpg-backed reminder persistence.

Two test surfaces:

* **SQL/arg shape** — mock the asyncpg pool and assert the impl issues
  the expected statements with the expected positional args. No real
  database needed. Always runs.
* **Live round-trip** — when ``LHA_TEST_POSTGRES_URL`` is set, run the
  same scenarios the in-memory store covers (add → list → claim_due →
  cancel) against a real Postgres. Skipped otherwise so CI stays
  hermetic.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

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


class _FakeConn:
    """Minimal stand-in for an asyncpg ``Connection`` inside a transaction."""

    def __init__(self, pool: _FakePool) -> None:
        self._pool = pool

    def transaction(self) -> Any:
        @asynccontextmanager
        async def _ctx():
            yield None

        return _ctx()

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self._pool.fetch_calls.append((query, args))
        if not self._pool.fetch_results:
            return []
        return self._pool.fetch_results.pop(0)

    async def execute(self, query: str, *args: Any) -> str:
        self._pool.execute_calls.append((query, args))
        return "OK"


class _FakePool:
    """Minimal stand-in for ``asyncpg.Pool`` recording every SQL call."""

    def __init__(self, fetch_results: list[list[Any]] | None = None):
        self.execute = AsyncMock()
        self.fetch_results = fetch_results or []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

        async def _execute(query: str, *args: Any) -> str:
            self.execute_calls.append((query, args))
            return "OK"

        async def _fetch(query: str, *args: Any) -> list[Any]:
            self.fetch_calls.append((query, args))
            if not self.fetch_results:
                return []
            return self.fetch_results.pop(0)

        self.execute = _execute  # type: ignore[assignment]
        self.fetch = _fetch  # type: ignore[assignment]

    def acquire(self) -> Any:
        @asynccontextmanager
        async def _ctx():
            yield _FakeConn(self)

        return _ctx()


@pytest.fixture
def fake_pool(monkeypatch):
    """Patch ``_ensure_pool`` on the postgres store to return a ``_FakePool``."""
    from horizon.scheduler import postgres_store as ps

    pool = _FakePool()

    async def _fake_ensure_pool(self):
        self._pool = pool
        self._schema_ready = True
        return pool

    monkeypatch.setattr(
        ps.PostgresReminderStore, "_ensure_pool", _fake_ensure_pool
    )
    return pool


# =============================================================================
# Module surface
# =============================================================================


async def test_module_exposes_postgres_store():
    from horizon.scheduler import postgres_store

    assert hasattr(postgres_store, "PostgresReminderStore")


async def test_factory_wires_postgres_backend(monkeypatch):
    from horizon.scheduler import postgres_store
    from horizon.scheduler import store as store_mod

    monkeypatch.setenv("LHA_REMINDER_STORE", "postgres")
    monkeypatch.setenv("LHA_REMINDER_DB_URL", "postgres://lha@localhost/lha")
    store_mod.reset_reminder_store()
    store = store_mod.get_reminder_store()
    assert isinstance(store, postgres_store.PostgresReminderStore)
    store_mod.reset_reminder_store()


async def test_factory_postgres_requires_url(monkeypatch):
    from horizon.scheduler import store as store_mod

    monkeypatch.setenv("LHA_REMINDER_STORE", "postgres")
    monkeypatch.delenv("LHA_REMINDER_DB_URL", raising=False)
    store_mod.reset_reminder_store()
    with pytest.raises(ValueError, match="LHA_REMINDER_DB_URL"):
        store_mod.get_reminder_store()
    store_mod.reset_reminder_store()


# =============================================================================
# SQL shape — mocked pool
# =============================================================================


class TestAddSqlShape:
    async def test_add_assigns_uuid_and_inserts_all_columns(self, fake_pool):
        from horizon.scheduler.postgres_store import PostgresReminderStore

        store = PostgresReminderStore(dsn="postgres://x")
        rid = await store.add(_reminder())

        assert isinstance(rid, str) and rid
        assert len(fake_pool.execute_calls) == 1
        query, args = fake_pool.execute_calls[0]
        assert "INSERT INTO reminders" in query
        # id is first, then all 8 dataclass fields → 9 positional args
        assert len(args) == 9
        assert args[0] == rid  # generated id
        assert args[1] == "u1"  # user_id

    async def test_add_uses_existing_id_when_provided(self, fake_pool):
        from horizon.scheduler.postgres_store import PostgresReminderStore
        from horizon.scheduler.store import Reminder

        store = PostgresReminderStore(dsn="postgres://x")
        explicit = Reminder(
            id="explicit-id",
            user_id="u1",
            app_name="lha",
            channel="cli",
            recipient_id="u1",
            message="x",
            fire_at=_at(2026, 5, 20, 9),
            recurrence=None,
            created_at=_at(2026, 5, 19, 12),
        )
        rid = await store.add(explicit)
        assert rid == "explicit-id"
        _, args = fake_pool.execute_calls[0]
        assert args[0] == "explicit-id"


class TestClaimDueSqlShape:
    async def test_claim_due_uses_skip_locked_and_deletes_one_shot(
        self, fake_pool
    ):
        from horizon.scheduler.postgres_store import PostgresReminderStore

        fake_pool.fetch_results.append(
            [
                {
                    "id": "r1",
                    "user_id": "u1",
                    "app_name": "lha",
                    "channel": "cli",
                    "recipient_id": "u1",
                    "message": "ping",
                    "fire_at": _at(2026, 5, 20, 8),
                    "recurrence": None,
                    "created_at": _at(2026, 5, 19, 12),
                }
            ]
        )
        store = PostgresReminderStore(dsn="postgres://x")
        cursor = _at(2026, 5, 20, 9)
        claimed = await store.claim_due(cursor)
        assert len(claimed) == 1
        assert claimed[0].id == "r1"
        query, args = fake_pool.fetch_calls[0]
        assert "FOR UPDATE SKIP LOCKED" in query
        assert args == (cursor,)
        # one-shot row should be DELETEd inside the same transaction
        delete_calls = [
            c for c in fake_pool.execute_calls if "DELETE" in c[0].upper()
        ]
        assert any(c[1] == ("r1",) for c in delete_calls)

    async def test_claim_due_advances_recurring(self, fake_pool):
        from horizon.scheduler.postgres_store import PostgresReminderStore

        fake_pool.fetch_results.append(
            [
                {
                    "id": "r1",
                    "user_id": "u1",
                    "app_name": "lha",
                    "channel": "cli",
                    "recipient_id": "u1",
                    "message": "ping",
                    "fire_at": _at(2026, 5, 20, 9),
                    "recurrence": "daily",
                    "created_at": _at(2026, 5, 19, 12),
                }
            ]
        )
        store = PostgresReminderStore(dsn="postgres://x")
        await store.claim_due(_at(2026, 5, 20, 9))
        # UPDATE should set fire_at to +1 day
        update_calls = [
            c for c in fake_pool.execute_calls if "UPDATE" in c[0].upper()
        ]
        assert len(update_calls) == 1
        next_at, rid = update_calls[0][1]
        assert rid == "r1"
        assert next_at == _at(2026, 5, 20, 9) + timedelta(days=1)

    async def test_claim_due_drops_rows_with_invalid_recurrence(
        self, fake_pool
    ):
        from horizon.scheduler.postgres_store import PostgresReminderStore

        fake_pool.fetch_results.append(
            [
                {
                    "id": "bad",
                    "user_id": "u1",
                    "app_name": "lha",
                    "channel": "cli",
                    "recipient_id": "u1",
                    "message": "ping",
                    "fire_at": _at(2026, 5, 20, 8),
                    "recurrence": "monthly",  # not in {daily,weekly,hourly,None}
                    "created_at": _at(2026, 5, 19, 12),
                }
            ]
        )
        store = PostgresReminderStore(dsn="postgres://x")
        claimed = await store.claim_due(_at(2026, 5, 20, 9))
        # Bad row is dropped, not returned; subsequent ticks won't keep crashing.
        assert claimed == []
        delete_calls = [
            c for c in fake_pool.execute_calls if "DELETE" in c[0].upper()
        ]
        assert any(c[1] == ("bad",) for c in delete_calls)


class TestListForUserSqlShape:
    async def test_list_for_user_scopes_by_user_id(self, fake_pool):
        from horizon.scheduler.postgres_store import PostgresReminderStore

        fake_pool.fetch_results.append([])
        store = PostgresReminderStore(dsn="postgres://x")
        await store.list_for_user("alice")
        query, args = fake_pool.fetch_calls[0]
        assert "user_id" in query
        assert args == ("alice",)


class TestCancelSqlShape:
    async def test_cancel_scopes_to_owner(self, fake_pool):
        from horizon.scheduler.postgres_store import PostgresReminderStore

        # asyncpg returns "DELETE N" — N=1 means a row was deleted
        async def _execute_one(query: str, *args: Any) -> str:
            fake_pool.execute_calls.append((query, args))
            return "DELETE 1"

        fake_pool.execute = _execute_one  # type: ignore[assignment]
        store = PostgresReminderStore(dsn="postgres://x")
        ok = await store.cancel("rid-1", "alice")
        assert ok is True
        query, args = fake_pool.execute_calls[0]
        assert "DELETE" in query.upper()
        assert "user_id" in query
        assert set(args) == {"rid-1", "alice"}

    async def test_cancel_returns_false_when_not_owned(self, fake_pool):
        from horizon.scheduler.postgres_store import PostgresReminderStore

        async def _execute_zero(query: str, *args: Any) -> str:
            fake_pool.execute_calls.append((query, args))
            return "DELETE 0"

        fake_pool.execute = _execute_zero  # type: ignore[assignment]
        store = PostgresReminderStore(dsn="postgres://x")
        ok = await store.cancel("rid-1", "bob")
        assert ok is False


# =============================================================================
# Schema bootstrap
# =============================================================================


class TestSchemaBootstrap:
    async def test_ensure_schema_runs_create_table_if_not_exists(self):
        """The DDL string should be idempotent for replica safety."""
        from horizon.scheduler.postgres_store import _SCHEMA_SQL

        assert "CREATE TABLE IF NOT EXISTS reminders" in _SCHEMA_SQL
        assert "CREATE INDEX IF NOT EXISTS" in _SCHEMA_SQL


# =============================================================================
# Live round-trip — skip unless real Postgres URL provided
# =============================================================================


_LIVE_URL = os.environ.get("LHA_TEST_POSTGRES_URL")
_skip_no_db = pytest.mark.skipif(
    not _LIVE_URL,
    reason="LHA_TEST_POSTGRES_URL not set; skipping live Postgres tests",
)


@_skip_no_db
class TestLiveRoundTrip:
    @pytest_asyncio.fixture
    async def live_store(self):
        from horizon.scheduler.postgres_store import PostgresReminderStore

        store = PostgresReminderStore(dsn=_LIVE_URL or "")
        # Wipe between tests
        pool = await store._ensure_pool()
        await pool.execute("DELETE FROM reminders")
        yield store
        await store.close()

    async def test_add_then_list(self, live_store):
        rid = await live_store.add(_reminder(user_id="alice"))
        items = await live_store.list_for_user("alice")
        assert len(items) == 1
        assert items[0].id == rid

    async def test_cancel_owner_scoped(self, live_store):
        rid = await live_store.add(_reminder(user_id="alice"))
        assert await live_store.cancel(rid, "bob") is False
        assert await live_store.cancel(rid, "alice") is True
        assert await live_store.list_for_user("alice") == []

    async def test_claim_due_round_trip_one_shot(self, live_store):
        rid = await live_store.add(
            _reminder(user_id="alice", fire_at=_at(2026, 5, 20, 8))
        )
        claimed = await live_store.claim_due(_at(2026, 5, 20, 9))
        assert [r.id for r in claimed] == [rid]
        # Second claim returns nothing — row was atomically removed
        assert await live_store.claim_due(_at(2026, 5, 20, 9)) == []
        assert await live_store.list_for_user("alice") == []

    async def test_claim_due_round_trip_recurring(self, live_store):
        rid = await live_store.add(
            _reminder(
                user_id="alice",
                fire_at=_at(2026, 5, 20, 9),
                recurrence="daily",
            )
        )
        claimed = await live_store.claim_due(_at(2026, 5, 20, 9))
        assert [r.id for r in claimed] == [rid]
        remaining = await live_store.list_for_user("alice")
        assert len(remaining) == 1
        assert remaining[0].fire_at == _at(2026, 5, 21, 9)
