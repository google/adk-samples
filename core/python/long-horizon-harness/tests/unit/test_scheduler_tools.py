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

"""``reminder(action='schedule'|'list'|'cancel')`` dispatch.

Surfaces validated:

* ``when`` accepts ISO 8601 and natural language ("tomorrow 9pm",
  "in 1 hour"). All resolved to UTC.
* Identity is pulled from ``tool_context._invocation_context``
  (``user_id`` / ``app_name``); channel + recipient_id are fixed to
  ``"cli"`` / ``user_id``.
* ``action='list'`` returns only the calling user's items.
* ``action='cancel'`` is user-scoped — user A cannot cancel user B's id.
* ``recurrence`` validation: ``None`` or one of ``daily|weekly|hourly``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.asyncio


def _ctx(
    *,
    user_id: str = "u1",
    app_name: str = "lha",
):
    return SimpleNamespace(
        _invocation_context=SimpleNamespace(user_id=user_id, app_name=app_name),
        state={},
    )


@pytest.fixture(autouse=True)
def _isolate_store(monkeypatch):
    """Each test gets a fresh InMemoryReminderStore singleton."""
    from horizon.scheduler import store as store_mod

    store_mod.reset_reminder_store()
    monkeypatch.delenv("LHA_REMINDER_STORE", raising=False)
    yield
    store_mod.reset_reminder_store()


# =============================================================================
# Module surface
# =============================================================================


async def test_module_exposes_dispatch_entrypoint():
    from horizon.scheduler import tools

    assert callable(tools.reminder)


# =============================================================================
# reminder(action='schedule')
# =============================================================================


class TestScheduleReminder:
    async def test_iso_when_round_trips(self):
        from horizon.scheduler.tools import reminder

        result = await reminder(
            action="schedule",
            when="2026-12-25T09:00:00Z",
            message="open presents",
            tool_context=_ctx(),
        )
        assert result["success"] is True
        assert result["fire_at"].startswith("2026-12-25T09:00:00")
        assert result["recurrence"] is None
        assert isinstance(result["id"], str) and result["id"]

    async def test_natural_language_when_resolves(self):
        from horizon.scheduler.tools import reminder

        result = await reminder(
            action="schedule",
            when="in 2 hours",
            message="check oven",
            tool_context=_ctx(),
        )
        assert result["success"] is True
        # We don't pin the exact resolved time (depends on clock); just
        # check that a usable ISO timestamp was produced.
        datetime.fromisoformat(result["fire_at"])

    async def test_unparseable_when_returns_error(self):
        from horizon.scheduler.tools import reminder

        result = await reminder(
            action="schedule",
            when="banana flapjack",
            message="x",
            tool_context=_ctx(),
        )
        assert result["success"] is False
        assert "when" in result["error"].lower()

    async def test_empty_message_rejected(self):
        from horizon.scheduler.tools import reminder

        result = await reminder(
            action="schedule",
            when="2026-12-25T09:00:00Z",
            message="   ",
            tool_context=_ctx(),
        )
        assert result["success"] is False

    async def test_recurrence_persisted(self):
        from horizon.scheduler.store import get_reminder_store
        from horizon.scheduler.tools import reminder

        result = await reminder(
            action="schedule",
            when="2026-12-25T09:00:00Z",
            message="standup",
            recurrence="daily",
            tool_context=_ctx(),
        )
        assert result["success"] is True
        items = await get_reminder_store().list_for_user("u1")
        assert len(items) == 1
        assert items[0].recurrence == "daily"

    async def test_invalid_recurrence_rejected(self):
        from horizon.scheduler.tools import reminder

        result = await reminder(
            action="schedule",
            when="2026-12-25T09:00:00Z",
            message="x",
            recurrence="yearly",
            tool_context=_ctx(),
        )
        assert result["success"] is False
        assert "recurrence" in result["error"].lower()

    async def test_channel_is_cli_and_recipient_is_user(self):
        from horizon.scheduler.store import get_reminder_store
        from horizon.scheduler.tools import reminder

        await reminder(
            action="schedule",
            when="2026-12-25T09:00:00Z",
            message="x",
            tool_context=_ctx(user_id="alice"),
        )
        items = await get_reminder_store().list_for_user("alice")
        assert items[0].channel == "cli"
        assert items[0].recipient_id == "alice"

    async def test_fire_at_is_utc(self):
        from horizon.scheduler.store import get_reminder_store
        from horizon.scheduler.tools import reminder

        await reminder(
            action="schedule",
            when="2026-12-25T09:00:00Z",
            message="x",
            tool_context=_ctx(),
        )
        items = await get_reminder_store().list_for_user("u1")
        assert items[0].fire_at.tzinfo is not None
        assert items[0].fire_at.utcoffset() == UTC.utcoffset(None)


# =============================================================================
# reminder(action='list' | 'cancel')
# =============================================================================


class TestListAndCancel:
    async def test_list_returns_only_caller(self):
        from horizon.scheduler.tools import reminder

        await reminder(
            action="schedule",
            when="2026-12-25T09:00:00Z",
            message="alice item",
            tool_context=_ctx(user_id="alice"),
        )
        await reminder(
            action="schedule",
            when="2026-12-25T09:00:00Z",
            message="bob item",
            tool_context=_ctx(user_id="bob"),
        )
        alice_view = await reminder(
            action="list", tool_context=_ctx(user_id="alice")
        )
        bob_view = await reminder(
            action="list", tool_context=_ctx(user_id="bob")
        )
        assert [r["message"] for r in alice_view["reminders"]] == ["alice item"]
        assert [r["message"] for r in bob_view["reminders"]] == ["bob item"]

    async def test_cancel_scoped_to_owner(self):
        from horizon.scheduler.tools import reminder

        scheduled = await reminder(
            action="schedule",
            when="2026-12-25T09:00:00Z",
            message="alice's",
            tool_context=_ctx(user_id="alice"),
        )
        rid = scheduled["id"]

        # bob cannot cancel
        bob_attempt = await reminder(
            action="cancel", id=rid, tool_context=_ctx(user_id="bob")
        )
        assert bob_attempt["success"] is False

        # alice can
        ok = await reminder(
            action="cancel", id=rid, tool_context=_ctx(user_id="alice")
        )
        assert ok["success"] is True

    async def test_cancel_unknown_returns_false(self):
        from horizon.scheduler.tools import reminder

        result = await reminder(
            action="cancel", id="not-real", tool_context=_ctx()
        )
        assert result["success"] is False


# =============================================================================
# Dispatch error branches
# =============================================================================


class TestDispatchErrors:
    async def test_schedule_missing_when_rejected(self):
        from horizon.scheduler.tools import reminder

        result = await reminder(
            action="schedule", message="x", tool_context=_ctx()
        )
        assert result["success"] is False
        assert "when" in result["error"].lower()

    async def test_schedule_missing_message_rejected(self):
        from horizon.scheduler.tools import reminder

        result = await reminder(
            action="schedule",
            when="2026-12-25T09:00:00Z",
            tool_context=_ctx(),
        )
        assert result["success"] is False
        assert "message" in result["error"].lower()

    async def test_cancel_missing_id_rejected(self):
        from horizon.scheduler.tools import reminder

        result = await reminder(action="cancel", tool_context=_ctx())
        assert result["success"] is False
        assert "id" in result["error"].lower()

    async def test_unknown_action_rejected(self):
        from horizon.scheduler.tools import reminder

        result = await reminder(action="banana", tool_context=_ctx())  # type: ignore[arg-type]
        assert result["success"] is False
        assert "action" in result["error"].lower()
