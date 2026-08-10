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

"""Reminder persistence behind a swappable ``ReminderStore`` Protocol.

The scheduler endpoints, the model-callable tools, and the Postgres-backed
impl all program against the Protocol; the in-memory impl works end-to-end
without a database.

A ``Reminder`` is one durable wake-up request: when to fire, who to
push to (channel + recipient_id captured at schedule time so the
out-of-band push reaches the right destination),
and an optional recurrence so the tick endpoint can re-schedule
instead of delete after firing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

Recurrence = Literal["daily", "weekly", "hourly"]
_VALID_RECURRENCES: tuple[str, ...] = ("daily", "weekly", "hourly")


@dataclass(frozen=True)
class Reminder:
    id: str
    user_id: str
    app_name: str
    channel: str
    recipient_id: str
    message: str
    fire_at: datetime
    recurrence: Recurrence | None
    created_at: datetime


def compute_next_fire_at(
    current: datetime, recurrence: Recurrence | None
) -> datetime | None:
    if recurrence is None:
        return None
    if recurrence == "daily":
        return current + timedelta(days=1)
    if recurrence == "weekly":
        return current + timedelta(days=7)
    if recurrence == "hourly":
        return current + timedelta(hours=1)
    raise ValueError(
        f"unknown recurrence {recurrence!r}; expected one of {_VALID_RECURRENCES}"
    )


class ReminderStore(Protocol):
    async def add(self, reminder: Reminder) -> str: ...

    async def claim_due(self, cursor: datetime) -> list[Reminder]:
        """Atomically advance/remove every due reminder and return what was claimed.

        Avoids duplicate firings when two ticks overlap.
        """
        ...

    async def list_for_user(self, user_id: str) -> list[Reminder]: ...

    async def cancel(self, reminder_id: str, user_id: str) -> bool: ...


class InMemoryReminderStore:
    """Dict-backed store; ``asyncio.Lock`` serializes mutations."""

    def __init__(self) -> None:
        self._rows: dict[str, Reminder] = {}
        self._lock = asyncio.Lock()

    async def add(self, reminder: Reminder) -> str:
        async with self._lock:
            rid = reminder.id or uuid.uuid4().hex
            stored = replace(reminder, id=rid)
            self._rows[rid] = stored
            return rid

    async def claim_due(self, cursor: datetime) -> list[Reminder]:
        async with self._lock:
            due = [r for r in self._rows.values() if r.fire_at <= cursor]
            claimed: list[Reminder] = []
            for r in due:
                try:
                    next_at = compute_next_fire_at(r.fire_at, r.recurrence)
                except ValueError:
                    logger.exception(
                        "claim_due: invalid recurrence %r on reminder %s; dropping",
                        r.recurrence,
                        r.id,
                    )
                    self._rows.pop(r.id, None)
                    continue
                if next_at is None:
                    self._rows.pop(r.id, None)
                else:
                    self._rows[r.id] = replace(r, fire_at=next_at)
                claimed.append(r)
            return claimed

    async def list_for_user(self, user_id: str) -> list[Reminder]:
        async with self._lock:
            return [r for r in self._rows.values() if r.user_id == user_id]

    async def cancel(self, reminder_id: str, user_id: str) -> bool:
        async with self._lock:
            existing = self._rows.get(reminder_id)
            if existing is None or existing.user_id != user_id:
                return False
            self._rows.pop(reminder_id, None)
            return True


_STORE_ENV = "LHA_REMINDER_STORE"
_singleton: ReminderStore | None = None


def get_reminder_store() -> ReminderStore:
    global _singleton
    if _singleton is not None:
        return _singleton
    backend = os.environ.get(_STORE_ENV, "memory").lower()
    if backend == "memory":
        _singleton = InMemoryReminderStore()
        return _singleton
    if backend == "postgres":
        from horizon.scheduler.postgres_store import build_from_env

        _singleton = build_from_env()
        return _singleton
    raise ValueError(
        f"unknown {_STORE_ENV}={backend!r}; expected 'memory' or 'postgres'"
    )


def reset_reminder_store() -> None:
    """Test hook: drop the cached singleton."""
    global _singleton
    _singleton = None


def active_reminder_store() -> ReminderStore | None:
    """Return the cached singleton, or None if it has not been built yet.

    Useful for shutdown hooks that want to close a Postgres pool only when
    one was actually opened.
    """
    return _singleton


__all__ = [
    "InMemoryReminderStore",
    "Recurrence",
    "Reminder",
    "ReminderStore",
    "active_reminder_store",
    "compute_next_fire_at",
    "get_reminder_store",
    "reset_reminder_store",
]
