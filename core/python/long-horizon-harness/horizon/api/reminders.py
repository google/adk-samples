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

"""/lha/reminders — per-user view + cancel of upcoming reminders.

Read/cancel surface over the same reminder store the agent's reminder()
tool uses; creating reminders stays with the agent.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException

from horizon.auth import current_user_id
from horizon.scheduler.store import get_reminder_store


def attach_reminders_routes(app: FastAPI) -> None:
    router = APIRouter(prefix="/lha", tags=["lha"])

    @router.get("/reminders")
    async def list_reminders(
        user_id: str = Depends(current_user_id),
    ) -> dict[str, Any]:
        reminders = sorted(
            await get_reminder_store().list_for_user(user_id),
            key=lambda r: r.fire_at,
        )
        return {
            "reminders": [
                {
                    "id": r.id,
                    "message": r.message,
                    "fire_at": r.fire_at.isoformat(),
                    "recurrence": r.recurrence,
                    "channel": r.channel,
                }
                for r in reminders
            ]
        }

    @router.delete("/reminders/{reminder_id}")
    async def cancel_reminder(
        reminder_id: str,
        user_id: str = Depends(current_user_id),
    ) -> dict[str, Any]:
        cancelled = await get_reminder_store().cancel(reminder_id, user_id)
        if not cancelled:
            raise HTTPException(status_code=404, detail="reminder not found")
        return {"id": reminder_id, "ok": True}

    app.include_router(router)
