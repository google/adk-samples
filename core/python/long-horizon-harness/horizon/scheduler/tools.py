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

"""Model-callable reminder tools.

A single ``reminder(action=...)`` dispatch over three operations:

* ``schedule``: parses natural language or ISO 8601 in ``when``, persists
  a row with the caller's identity and channel metadata, returns the new id.
* ``list``: scoped to the caller's ``user_id``.
* ``cancel``: same scoping, so user A cannot remove user B's reminder
  even if they guess the id.

A row's ``channel`` / ``recipient_id`` are the out-of-band push address
(``horizon/tools/push_to_user.py``). No inbound gateway exists — alternate
gateways are out of scope — so they are fixed to ``"cli"`` / ``user_id``;
delivery the user actually sees is the tagged session the tick creates.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal

from horizon.scheduler.store import (
    _VALID_RECURRENCES,
    Reminder,
    get_reminder_store,
)

logger = logging.getLogger(__name__)

_CHANNEL = "cli"


def _parse_when(value: str) -> datetime | None:
    """Return a UTC-aware datetime, or ``None`` if unparseable."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None

    if parsed is None:
        try:
            import dateparser
        except ImportError:
            return None
        parsed = dateparser.parse(
            text,
            settings={
                "PREFER_DATES_FROM": "future",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "TIMEZONE": "UTC",
                "TO_TIMEZONE": "UTC",
            },
        )

    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _identity(tool_context: Any) -> tuple[str, str] | None:
    invocation = getattr(tool_context, "_invocation_context", None)
    if invocation is None:
        return None
    user_id = getattr(invocation, "user_id", "") or ""
    app_name = getattr(invocation, "app_name", "") or ""
    if not user_id or not app_name:
        return None
    return user_id, app_name


async def _schedule(
    when: str,
    message: str,
    recurrence: str | None,
    tool_context: Any,
) -> dict[str, Any]:
    if not message or not message.strip():
        return {"success": False, "error": "message cannot be empty"}

    if recurrence is not None and recurrence not in _VALID_RECURRENCES:
        return {
            "success": False,
            "error": (
                f"invalid recurrence {recurrence!r}; expected one of "
                f"{_VALID_RECURRENCES} or None"
            ),
        }

    fire_at = _parse_when(when)
    if fire_at is None:
        return {
            "success": False,
            "error": (
                f"could not parse when={when!r}; pass ISO 8601 "
                "(2026-12-25T09:00:00Z) or natural language "
                "('tomorrow 9pm', 'in 1 hour')"
            ),
        }

    identity = _identity(tool_context)
    if identity is None:
        return {"success": False, "error": "no active invocation context"}
    user_id, app_name = identity

    store = get_reminder_store()
    now = datetime.now(UTC)
    new = Reminder(
        id="",
        user_id=user_id,
        app_name=app_name,
        channel=_CHANNEL,
        recipient_id=user_id,
        message=message.strip(),
        fire_at=fire_at,
        recurrence=recurrence,  # type: ignore[arg-type]
        created_at=now,
    )
    rid = await store.add(new)
    return {
        "success": True,
        "id": rid,
        "fire_at": fire_at.isoformat(),
        "recurrence": recurrence,
        "channel": _CHANNEL,
    }


async def _list(tool_context: Any) -> dict[str, Any]:
    identity = _identity(tool_context)
    if identity is None:
        return {"success": False, "error": "no active invocation context"}
    user_id, _app_name = identity
    store = get_reminder_store()
    items = await store.list_for_user(user_id)
    return {
        "success": True,
        "count": len(items),
        "reminders": [
            {
                "id": r.id,
                "message": r.message,
                "fire_at": r.fire_at.isoformat(),
                "recurrence": r.recurrence,
                "channel": r.channel,
            }
            for r in items
        ],
    }


async def _cancel(id: str, tool_context: Any) -> dict[str, Any]:
    if not id:
        return {"success": False, "error": "id is required"}
    identity = _identity(tool_context)
    if identity is None:
        return {"success": False, "error": "no active invocation context"}
    user_id, _app_name = identity
    store = get_reminder_store()
    ok = await store.cancel(id, user_id)
    if not ok:
        return {
            "success": False,
            "error": "no reminder with that id owned by you",
        }
    return {"success": True, "id": id}


async def reminder(
    action: Literal["schedule", "list", "cancel"],
    *,
    when: str | None = None,
    message: str | None = None,
    recurrence: str | None = None,
    id: str | None = None,
    tool_context: Any,
) -> dict[str, Any]:
    """Schedule, list, or cancel out-of-band reminders. ``action`` picks
    the operation:

    - ``schedule``: schedule a reminder for the caller. Requires ``when``
      (ISO 8601 like ``2026-12-25T09:00:00Z`` or natural language such as
      ``tomorrow 9pm`` / ``in 1 hour``, resolved to UTC) and ``message``
      (text delivered at fire time). Optional ``recurrence`` is one of
      ``daily``, ``weekly``, ``hourly``; omit for one-shot.
    - ``list``: return the caller's pending reminders. No other args.
    - ``cancel``: cancel one of the caller's reminders. Requires ``id``
      (from ``schedule`` or ``list``). A reminder owned by another user
      cannot be cancelled.
    """
    if action == "schedule":
        if when is None:
            return {
                "success": False,
                "error": "action='schedule' requires 'when'",
            }
        if message is None:
            return {
                "success": False,
                "error": "action='schedule' requires 'message'",
            }
        return await _schedule(when, message, recurrence, tool_context)
    if action == "list":
        return await _list(tool_context)
    if action == "cancel":
        if id is None:
            return {"success": False, "error": "action='cancel' requires 'id'"}
        return await _cancel(id, tool_context)
    return {
        "success": False,
        "error": f"Unknown action {action!r}. Use: schedule, list, cancel.",
    }


__all__ = ["reminder"]
