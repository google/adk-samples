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

"""Lifespan-teardown ordering for the FastAPI app.

The teardown must drain in-flight sibling-agent tasks BEFORE closing the
reminder store, because siblings can still be writing reminders when SIGTERM
arrives. Either close() raising must not block the other.
"""

from __future__ import annotations

import os
import sys
import types

# Skip the module-level FastAPI build (which calls google.auth.default() and
# hits Agent Engine) before importing the module.
os.environ["LHA_ADK_SKIP_APP_BUILD"] = "true"

import pytest

from horizon import fast_api_app


class _FakePlugin:
    def __init__(self, *, raises: bool = False) -> None:
        self.closed = False
        self.raises = raises

    async def close(self) -> None:
        self.closed = True
        if self.raises:
            raise RuntimeError("simulated sibling drain failure")


class _FakeStore:
    def __init__(self, *, raises: bool = False) -> None:
        self.closed = False
        self.raises = raises

    async def close(self) -> None:
        self.closed = True
        if self.raises:
            raise RuntimeError("simulated reminder close failure")


def _install_fake_plugin(
    monkeypatch: pytest.MonkeyPatch, plugin: _FakePlugin
) -> None:
    fake_agent = types.ModuleType("horizon.agent")
    fake_agent.SIBLING_AGENT_PLUGIN = plugin  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "horizon.agent", fake_agent)


@pytest.mark.asyncio
async def test_shutdown_drains_sibling_plugin_then_reminder_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    plugin = _FakePlugin()
    store = _FakeStore()

    async def _plugin_close() -> None:
        order.append("plugin")

    async def _store_close() -> None:
        order.append("store")

    plugin.close = _plugin_close  # type: ignore[method-assign]
    store.close = _store_close  # type: ignore[method-assign]
    _install_fake_plugin(monkeypatch, plugin)
    monkeypatch.setattr(
        fast_api_app.reminder_store, "active_reminder_store", lambda: store
    )

    await fast_api_app._shutdown_runtime()

    assert order == ["plugin", "store"]


@pytest.mark.asyncio
async def test_shutdown_continues_when_sibling_close_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _FakePlugin(raises=True)
    store = _FakeStore()
    _install_fake_plugin(monkeypatch, plugin)
    monkeypatch.setattr(
        fast_api_app.reminder_store, "active_reminder_store", lambda: store
    )

    await fast_api_app._shutdown_runtime()

    assert plugin.closed
    assert store.closed


@pytest.mark.asyncio
async def test_shutdown_continues_when_reminder_close_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _FakePlugin()
    store = _FakeStore(raises=True)
    _install_fake_plugin(monkeypatch, plugin)
    monkeypatch.setattr(
        fast_api_app.reminder_store, "active_reminder_store", lambda: store
    )

    await fast_api_app._shutdown_runtime()

    assert plugin.closed
    assert store.closed


@pytest.mark.asyncio
async def test_shutdown_skips_reminder_close_when_store_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _FakePlugin()
    _install_fake_plugin(monkeypatch, plugin)
    monkeypatch.setattr(
        fast_api_app.reminder_store, "active_reminder_store", lambda: None
    )
    monkeypatch.setattr(
        "horizon.scheduler.routine_store.active_routine_store", lambda: None
    )

    await fast_api_app._shutdown_runtime()

    assert plugin.closed


@pytest.mark.asyncio
async def test_shutdown_closes_routine_store_after_reminder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    plugin = _FakePlugin()
    _install_fake_plugin(monkeypatch, plugin)
    reminder, routine = _FakeStore(), _FakeStore()

    async def _rem_close() -> None:
        order.append("reminder")

    async def _rout_close() -> None:
        order.append("routine")

    reminder.close = _rem_close  # type: ignore[method-assign]
    routine.close = _rout_close  # type: ignore[method-assign]
    monkeypatch.setattr(
        fast_api_app.reminder_store, "active_reminder_store", lambda: reminder
    )
    monkeypatch.setattr(
        "horizon.scheduler.routine_store.active_routine_store", lambda: routine
    )

    await fast_api_app._shutdown_runtime()

    assert order == ["reminder", "routine"]


@pytest.mark.asyncio
async def test_shutdown_continues_when_routine_close_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _FakePlugin()
    _install_fake_plugin(monkeypatch, plugin)
    reminder = _FakeStore()
    routine = _FakeStore(raises=True)
    monkeypatch.setattr(
        fast_api_app.reminder_store, "active_reminder_store", lambda: reminder
    )
    monkeypatch.setattr(
        "horizon.scheduler.routine_store.active_routine_store", lambda: routine
    )

    await fast_api_app._shutdown_runtime()

    assert reminder.closed
    assert routine.closed
