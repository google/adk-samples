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

"""``PostgresReminderStore`` — asyncpg-backed implementation of the
:class:`~horizon.scheduler.store.ReminderStore` protocol.

Connection string is read from ``LHA_REMINDER_DB_URL`` (set in Cloud
Run by Terraform). The schema bootstrap (``CREATE TABLE IF NOT EXISTS``)
runs once per process on first pool use — idempotent and safe across
Cloud Run replicas because Postgres serializes the DDL.

We deliberately skip a full migration framework (yoyo) for one table.
When the schema needs to evolve, swap this for migrations.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

import asyncpg

from horizon.infrastructure.db_resilience import retry_on_disconnect
from horizon.scheduler.store import Reminder, compute_next_fire_at

logger = logging.getLogger(__name__)

# Recycle idle asyncpg connections well under Cloud SQL's idle timeout so the
# every-minute tick never hands out a connection the server already reaped.
_MAX_INACTIVE_LIFETIME_SECONDS = 300.0
_CONNECT_TIMEOUT_SECONDS = 10.0


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS reminders (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    app_name      TEXT NOT NULL,
    channel       TEXT NOT NULL,
    recipient_id  TEXT NOT NULL,
    message       TEXT NOT NULL,
    fire_at       TIMESTAMPTZ NOT NULL,
    recurrence    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS reminders_fire_at_idx ON reminders (fire_at);
CREATE INDEX IF NOT EXISTS reminders_user_id_idx ON reminders (user_id);
"""


_INSERT_SQL = """
INSERT INTO reminders (
    id, user_id, app_name, channel, recipient_id,
    message, fire_at, recurrence, created_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
"""

# FOR UPDATE SKIP LOCKED so two concurrent ticks claim disjoint rows.
_CLAIM_DUE_SQL = """
SELECT id, user_id, app_name, channel, recipient_id,
       message, fire_at, recurrence, created_at
FROM reminders
WHERE fire_at <= $1
ORDER BY fire_at ASC
FOR UPDATE SKIP LOCKED
"""

_SELECT_USER_SQL = """
SELECT id, user_id, app_name, channel, recipient_id,
       message, fire_at, recurrence, created_at
FROM reminders
WHERE user_id = $1
ORDER BY fire_at ASC
"""

_DELETE_SQL = "DELETE FROM reminders WHERE id = $1"
_UPDATE_FIRE_AT_SQL = "UPDATE reminders SET fire_at = $1 WHERE id = $2"
_CANCEL_SQL = "DELETE FROM reminders WHERE id = $1 AND user_id = $2"


class PostgresReminderStore:
    """asyncpg-backed reminder store. Pool + schema initialized lazily."""

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("PostgresReminderStore requires a non-empty DSN")
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._schema_ready = False
        self._init_lock = asyncio.Lock()
        self._retry_base_delay = 0.25

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is not None and self._schema_ready:
            return self._pool
        async with self._init_lock:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=1,
                    # Bounded to stay within the per-instance connection budget
                    # (see db_resilience): the scheduler is low-QPS.
                    max_size=3,
                    max_inactive_connection_lifetime=_MAX_INACTIVE_LIFETIME_SECONDS,
                    timeout=_CONNECT_TIMEOUT_SECONDS,
                )
            if not self._schema_ready:
                await self._pool.execute(_SCHEMA_SQL)
                self._schema_ready = True
            return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._schema_ready = False

    async def add(self, reminder: Reminder) -> str:
        pool = await self._ensure_pool()
        rid = reminder.id or uuid.uuid4().hex

        async def op():
            await pool.execute(
                _INSERT_SQL,
                rid,
                reminder.user_id,
                reminder.app_name,
                reminder.channel,
                reminder.recipient_id,
                reminder.message,
                reminder.fire_at,
                reminder.recurrence,
                reminder.created_at,
            )

        await retry_on_disconnect(op, base_delay=self._retry_base_delay)
        return rid

    async def claim_due(self, cursor: Any) -> list[Reminder]:
        pool = await self._ensure_pool()

        # Wrap the whole acquire/transaction so a mid-claim drop re-runs the
        # entire claim atomically — a partial claim must not be retried piecemeal.
        async def op() -> list[Reminder]:
            claimed: list[Reminder] = []
            async with pool.acquire() as conn:
                async with conn.transaction():
                    rows = await conn.fetch(_CLAIM_DUE_SQL, cursor)
                    for row in rows:
                        reminder = _row_to_reminder(row)
                        try:
                            next_at = compute_next_fire_at(
                                reminder.fire_at, reminder.recurrence
                            )
                        except ValueError:
                            logger.exception(
                                "claim_due: invalid recurrence %r on reminder %s; "
                                "dropping",
                                reminder.recurrence,
                                reminder.id,
                            )
                            await conn.execute(_DELETE_SQL, reminder.id)
                            continue
                        if next_at is None:
                            await conn.execute(_DELETE_SQL, reminder.id)
                        else:
                            await conn.execute(
                                _UPDATE_FIRE_AT_SQL, next_at, reminder.id
                            )
                        claimed.append(reminder)
            return claimed

        return await retry_on_disconnect(op, base_delay=self._retry_base_delay)

    async def list_for_user(self, user_id: str) -> list[Reminder]:
        pool = await self._ensure_pool()

        async def op():
            return await pool.fetch(_SELECT_USER_SQL, user_id)

        rows = await retry_on_disconnect(op, base_delay=self._retry_base_delay)
        return [_row_to_reminder(r) for r in rows]

    async def cancel(self, reminder_id: str, user_id: str) -> bool:
        pool = await self._ensure_pool()

        async def op():
            return await pool.execute(_CANCEL_SQL, reminder_id, user_id)

        status = await retry_on_disconnect(
            op, base_delay=self._retry_base_delay
        )
        # asyncpg returns "DELETE N" — N==0 means no row matched
        return status.upper().startswith("DELETE ") and not status.endswith(
            " 0"
        )


def _row_to_reminder(row: Any) -> Reminder:
    return Reminder(
        id=row["id"],
        user_id=row["user_id"],
        app_name=row["app_name"],
        channel=row["channel"],
        recipient_id=row["recipient_id"],
        message=row["message"],
        fire_at=row["fire_at"],
        recurrence=row["recurrence"],
        created_at=row["created_at"],
    )


def build_from_env() -> PostgresReminderStore:
    """Construct a store from ``LHA_REMINDER_DB_URL``.

    Raises ``ValueError`` if the env var is unset — surfaces config gaps
    at process startup instead of at first reminder write.
    """
    dsn = os.environ.get("LHA_REMINDER_DB_URL", "").strip()
    if not dsn:
        raise ValueError(
            "LHA_REMINDER_STORE=postgres requires LHA_REMINDER_DB_URL "
            "(e.g. postgres://user@/db?host=/cloudsql/<conn_name>)"
        )
    return PostgresReminderStore(dsn=dsn)


__all__ = ["PostgresReminderStore", "build_from_env"]
