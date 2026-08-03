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
"""report_to_maintainers — the agent's HITL-gated self-report channel."""

from __future__ import annotations

from typing import Any, Literal, get_args

from horizon.feedback.context import build_redacted_context
from horizon.feedback.sink import emit_feedback_record
from horizon.tools._hitl import request_user_confirmation

Category = Literal[
    "bug",
    "capability_gap",
    "confusing_instruction",
    "guardrail_misfire",
    "other",
]
CATEGORIES: tuple[str, ...] = get_args(Category)
SIGNATURES_KEY = "reported_signatures"


def _signature(category: str, summary: str) -> str:
    return f"{category}|{summary.strip().lower()}"


def report_to_maintainers(
    category: Category,
    summary: str,
    details: str,
    include_context: bool = False,
    *,
    tool_context: Any,
) -> dict[str, Any]:
    """Send feedback about Horizon itself to the project maintainers (asks first).

    Use to self-report a problem with the agent: a tool that errored
    repeatedly, a guardrail that misfired, a missing capability, or a
    confusing instruction. The user is always shown the draft and must
    confirm before anything is sent.

    Args:
        category: One of bug, capability_gap, confusing_instruction,
            guardrail_misfire, other.
        summary: One-line title of the issue.
        details: What happened and what you expected instead.
        include_context: Attach a redacted summary of this conversation.
            Only set when the conversation itself is the evidence.
    """
    if category not in CATEGORIES:
        return {
            "success": False,
            "error": f"category must be one of {', '.join(CATEGORIES)}.",
        }
    if not summary or not summary.strip():
        return {"success": False, "error": "summary is required."}
    if not details or not details.strip():
        return {"success": False, "error": "details are required."}

    summary = summary.strip()
    details = details.strip()
    signature = _signature(category, summary)

    state = getattr(tool_context, "state", None)
    reported = (
        list(state.get(SIGNATURES_KEY) or []) if state is not None else []
    )
    if signature in reported:
        return {
            "success": True,
            "status": "already_reported",
            "summary": summary,
        }

    draft = f"[{category}] {summary}\n\n{details}"
    if include_context:
        draft += (
            "\n\n(A redacted summary of this conversation will be attached.)"
        )

    tc = request_user_confirmation(
        tool_context,
        hint=draft,
        payload={
            "category": category,
            "summary": summary,
            "details": details,
            "include_context": include_context,
        },
    )
    if tc is None:
        return {
            "success": True,
            "status": "awaiting_user_response",
            "summary": summary,
        }
    if not getattr(tc, "confirmed", False):
        return {
            "success": False,
            "status": "declined",
            "error": "User declined to send the report.",
        }

    inv = getattr(tool_context, "_invocation_context", None)
    session = getattr(inv, "session", None)
    record: dict[str, Any] = {
        "log_type": "feedback",
        "service_name": "lha",
        "origin": "agent",
        "category": category,
        "text": f"{summary}\n\n{details}",
        "session_id": getattr(session, "id", None),
        "user_id": getattr(inv, "user_id", None),
    }
    if include_context and session is not None:
        record["context_summary"] = build_redacted_context(session)

    try:
        emit_feedback_record(record)
    except Exception:
        return {
            "success": False,
            "status": "send_failed",
            "error": "Could not send the report — please tell the user it failed.",
        }

    if state is not None:
        state[SIGNATURES_KEY] = [*reported, signature]
    return {"success": True, "status": "filed", "summary": summary}
