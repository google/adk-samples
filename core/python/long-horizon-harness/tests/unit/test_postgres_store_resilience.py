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

"""Unit coverage for the asyncpg reminder-pool resilience (recycle + connect
timeout + retry-on-disconnect)."""

from __future__ import annotations

import asyncpg
import pytest

from horizon.scheduler.postgres_store import PostgresReminderStore


@pytest.mark.asyncio
async def test_create_pool_uses_resilient_kwargs(monkeypatch):
    captured = {}

    async def fake_create_pool(**kwargs):
        captured.update(kwargs)

        class _Pool:
            async def execute(self, *a, **k):
                return "OK"

        return _Pool()

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    store = PostgresReminderStore(dsn="postgres://u@/db?host=/cloudsql/x")
    await store._ensure_pool()

    assert captured["max_inactive_connection_lifetime"] == 300
    assert captured["timeout"] == 10
    # Bounded so the scheduler pool stays within the per-instance connection budget.
    assert captured["min_size"] == 1
    assert captured["max_size"] == 3


@pytest.mark.asyncio
async def test_list_for_user_retries_on_disconnect(monkeypatch):
    calls = {"n": 0}

    class _Pool:
        async def execute(self, *a, **k):
            return "OK"

        async def fetch(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise asyncpg.exceptions.ConnectionDoesNotExistError()
            return []

    async def fake_create_pool(**kwargs):
        return _Pool()

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    store = PostgresReminderStore(dsn="postgres://u@/db?host=/cloudsql/x")
    store._retry_base_delay = 0  # keep the test fast
    rows = await store.list_for_user("u1")
    assert rows == []
    assert calls["n"] == 2  # retried once
