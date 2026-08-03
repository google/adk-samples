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

"""add_memory tool: write durable facts into ADK Memory Bank.

Supports an "agent" vs "user" scope split; ADK Memory Bank owns persistence
(there is no on-disk store).
"""

from __future__ import annotations

from typing import Any, Literal

from google.adk.memory import BaseMemoryService
from google.adk.tools.tool_context import ToolContext

from horizon.memory._content_safety import (
    MEMORY_CHAR_LIMIT,
    USER_CHAR_LIMIT,
    scan_memory_content,
)
from horizon.memory._writer import entry_exists, write_memory_event
from horizon.telemetry.ui import record_memory_write

Scope = Literal["agent", "user"]
_VALID_SCOPES: tuple[str, ...] = ("agent", "user")
_SCOPE_PREFIX = {"agent": "[agent] ", "user": "[user] "}


def _char_limit(scope: Scope) -> int:
    if scope == "user":
        return USER_CHAR_LIMIT
    return MEMORY_CHAR_LIMIT


def _scoped_text(scope: Scope, content: str) -> str:
    return f"{_SCOPE_PREFIX[scope]}{content}"


async def add_memory_entry(
    *,
    content: str,
    scope: str,
    memory_service: BaseMemoryService,
    app_name: str,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    """Append one durable fact to the configured Memory Bank.

    Pure helper: takes the service + identifiers directly so it can be unit-
    tested without a Runner / InvocationContext. The LLM-callable surface is
    `add_memory` below, which pulls the same identifiers off ToolContext.
    """
    if scope not in _VALID_SCOPES:
        return {
            "success": False,
            "error": (
                f"Invalid scope {scope!r}. Use one of: {', '.join(_VALID_SCOPES)}."
            ),
        }

    cleaned = content.strip()
    if not cleaned:
        return {"success": False, "error": "Content cannot be empty."}

    limit = _char_limit(scope)
    if len(cleaned) > limit:
        return {
            "success": False,
            "error": (
                f"Content is {len(cleaned)} chars; {scope} scope limit is "
                f"{limit}. Shorten the entry or split it across multiple "
                "writes."
            ),
        }

    scan_error = scan_memory_content(cleaned)
    if scan_error:
        return {"success": False, "error": scan_error}

    scoped_text = _scoped_text(scope, cleaned)
    if await entry_exists(
        memory_service=memory_service,
        app_name=app_name,
        user_id=user_id,
        text=scoped_text,
    ):
        return {
            "success": True,
            "scope": scope,
            "message": "Entry already in memory (no duplicate added).",
        }

    await write_memory_event(
        text=scoped_text,
        scope=scope,
        invocation_id=f"add_memory:{scope}",
        author="user" if scope == "user" else "system",
        memory_service=memory_service,
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    return {"success": True, "scope": scope, "message": "Entry added."}


async def add_memory(
    content: str,
    scope: str = "agent",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Save a durable fact across sessions.

    Use proactively when you learn something that will still matter next
    session: user preferences, stable environment facts, project conventions,
    or corrections the user explicitly asks you to remember. Skip task
    progress, completion logs, or anything trivially re-derivable.

    Args:
        content: The fact to remember. Keep it compact and self-contained.
        scope: "user" for facts about who the user is (name, role,
            preferences, communication style); "agent" for your own notes
            (environment, conventions, tool quirks, lessons). Defaults to
            "agent".
    """
    if tool_context is None:
        return {
            "success": False,
            "error": "add_memory must be called via the agent runtime.",
        }

    invocation_context = tool_context._invocation_context
    if invocation_context.memory_service is None:
        return {
            "success": False,
            "error": "Memory service is not configured for this runtime.",
        }

    result = await add_memory_entry(
        content=content,
        scope=scope,
        memory_service=invocation_context.memory_service,
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
    )
    if result.get("success"):
        record_memory_write(
            tool_context=tool_context, scope=scope, content=content
        )
    return result
