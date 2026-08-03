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

"""``POST /scheduler/tick`` — fires every minute (Cloud Scheduler in prod).

Each call atomically claims every reminder whose ``fire_at`` is in the
past (advancing recurring rows and removing one-shots in the same
transaction), then pushes each via the a2a-gateway. Using an atomic
claim — rather than read-then-update — keeps two overlapping ticks
from delivering the same reminder twice.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from a2a.server.context import ServerCallContext
from a2a.types import Message, Part, SendMessageRequest, Task
from fastapi import APIRouter, Depends, Request
from google.adk.a2a import _compat as a2a_compat

from horizon.api.sessions import TITLE_MAX_LEN
from horizon.auth import user_identity_scope
from horizon.scheduler.auth import verify_cloud_scheduler_token
from horizon.scheduler.sessions import create_scheduled_session
from horizon.scheduler.store import Reminder, get_reminder_store
from horizon.tools.push_to_user import push_to_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/scheduler",
    dependencies=[Depends(verify_cloud_scheduler_token)],
)


def _text_from_parts(parts: list[Part] | None) -> str:
    out: list[str] = []
    for part in parts or []:
        if a2a_compat.is_text_part(part):
            text = a2a_compat.part_text(part).strip()
            if text:
                out.append(text)
    return "\n".join(out)


def _final_text_from_result(result: Message | Task) -> str:
    """Pull the agent's final reply text out of a Message or Task result."""
    if isinstance(result, Message):
        return _text_from_parts(result.parts)
    status_message = getattr(result.status, "message", None)
    if status_message is not None:
        text = _text_from_parts(status_message.parts)
        if text:
            return text
    for message in reversed(result.history or []):
        if message.role == a2a_compat.ROLE_AGENT:
            text = _text_from_parts(message.parts)
            if text:
                return text
    for artifact in result.artifacts or []:
        text = _text_from_parts(artifact.parts)
        if text:
            return text
    return ""


async def _run_reminder_turn(
    handler: Any, *, user_id: str, session_id: str, message: str
) -> str:
    """Run one agent turn for a fired reminder via the shared A2A handler.

    Targets the pre-tagged session by ``context_id`` and runs under the
    reminder's ``user_id``, so the executor records a Task the web UI can
    render — identical to a normal chat. Module-level hook so unit tests can
    stub it without driving an LLM.
    """
    params = SendMessageRequest(
        message=a2a_compat.make_message(
            message_id=uuid4().hex,
            role=a2a_compat.ROLE_USER,
            parts=[a2a_compat.make_text_part(message)],
            context_id=session_id,
        )
    )
    with user_identity_scope(user_id):
        result = await handler.on_message_send(params, ServerCallContext())
    return _final_text_from_result(result)


async def _fire_one(
    runner: Any, handler: Any, reminder: Reminder
) -> dict[str, Any]:
    """Create a tagged session, run the turn, then best-effort deliver."""
    text = reminder.message
    if runner is not None and handler is not None:
        try:
            session = await create_scheduled_session(
                runner.session_service,
                # Sessions are keyed under the runner's app_name; the executor
                # reattaches there, so the tag must land there too (matches
                # dream_review_endpoint). reminder.app_name is the fallback.
                app_name=getattr(runner, "app_name", None) or reminder.app_name,
                user_id=reminder.user_id,
                job_type="reminder",
                title=f"Reminder: {reminder.message}"[:TITLE_MAX_LEN],
            )
            result_text = await _run_reminder_turn(
                handler,
                user_id=reminder.user_id,
                session_id=session.id,
                message=reminder.message,
            )
            if result_text:
                text = result_text
        except Exception:
            logger.exception("tick: reminder turn failed for %s", reminder.id)

    # Delivery is decoupled: a missing/failing gateway must not fail the tick.
    try:
        return await push_to_user(
            user_id=reminder.user_id,
            app_name=reminder.app_name,
            channel=reminder.channel,
            recipient_id=reminder.recipient_id,
            text=text,
        )
    except Exception:
        logger.exception("tick: push raised for reminder %s", reminder.id)
        return {"delivered": False, "error": "exception"}


@router.post("/tick")
async def tick(request: Request) -> dict[str, Any]:
    """Fire every reminder due as of now."""
    store = get_reminder_store()
    runner = getattr(request.app.state, "runner", None)
    handler = getattr(request.app.state, "a2a_handler", None)
    cursor = datetime.now(UTC)
    claimed = await store.claim_due(cursor)
    fired = 0
    failed = 0

    # Reminders fire serially: each runs a full agent turn, so a tick with many due reminders is a long request (acceptable at low volume).
    for reminder in claimed:
        result = await _fire_one(runner, handler, reminder)
        if not result.get("delivered"):
            failed += 1
        fired += 1

    body: dict[str, Any] = {"fired": fired}
    if failed:
        body["failed"] = failed
    return body


__all__ = ["router"]
