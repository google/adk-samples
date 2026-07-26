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

"""Synthetic <system-reminder> builders for the rolling message tail.

These produce small, individually-tagged reminder strings that a
before_model_callback appends to llm_request.contents (NOT the cached
system_instruction). Keeping them out of the system prefix means the
stable prefix stays byte-identical across turns, so the model's
prompt/context cache keeps hitting; per-turn steering rides in the
tail where re-caching is cheap.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from horizon.memory.user_profile import render_user_profile
from horizon.tools.todos import read_todos, render_todos, session_id_of
from horizon.workspace_window import render_window, window_dirs

REMINDER_OPEN = "<system-reminder>"
REMINDER_CLOSE = "</system-reminder>"

_DATE_FORMAT = "%A, %B %d, %Y"
_DEFAULT_WARN_AT_REMAINING = 5
_MAX_ERROR_CHARS = 800

# Matches IterationBudgetPlugin's default max_iterations in horizon/agent.py.
_DEFAULT_MAX_ITERATIONS = 50

# Handoff flags owned by the two halt consumers (guardrails + budget). Held as
# literals so this hot-path module stays import-light; kept in sync with
# horizon/guardrails/halt_consumer.py and horizon/conversation/iteration_budget_plugin.py.
_GUARDRAILS_HANDOFF_DELIVERED_STATE_KEY = "__halt_handoff_delivered__"
_BUDGET_HANDOFF_DELIVERED_STATE_KEY = "_iteration_budget_handoff_delivered"


def _wrap(body: str) -> str:
    return f"{REMINDER_OPEN}\n{body.strip()}\n{REMINDER_CLOSE}"


def build_budget_reminder(
    *,
    iteration: int,
    max_iterations: int,
    warn_at_remaining: int = _DEFAULT_WARN_AT_REMAINING,
) -> str | None:
    if max_iterations <= 0:
        return None
    remaining = max_iterations - iteration
    if remaining > warn_at_remaining:
        return None
    remaining = max(remaining, 0)
    return _wrap(
        f"Iteration budget running low: {remaining} of {max_iterations} "
        "iterations remain before this turn is force-stopped. Wrap up — "
        "finish the current step, then deliver your result rather than "
        "starting new long-running work."
    )


def build_error_reminder(*, last_error: str | None) -> str | None:
    if not last_error:
        return None
    text = str(last_error)
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS] + " […truncated]"
    return _wrap(
        f"Your last tool call failed: {text}\n"
        "Before retrying, diagnose the cause — read the failing file or its "
        "test, check what changed. Do not re-run the same call with a tweaked "
        "flag; that is a retry, not a fix."
    )


def build_volatile_reminder(
    *,
    state: dict[str, Any] | None = None,
    always_include_date: bool = True,
) -> str | None:
    state = state or {}
    lines: list[str] = []

    profile = render_user_profile(state)
    if profile:
        lines.append(profile)

    focus = render_window(window_dirs(state))
    if focus:
        lines.append(focus)

    iteration = state.get("iteration") or 0
    if iteration:
        lines.append(f"Iteration: {iteration}")

    # last_error is surfaced once, via build_error_reminder (truncated + diagnosis
    # guidance); don't duplicate the (untruncated) error in the volatile tail.
    if always_include_date or lines:
        lines.append(f"Today is {datetime.now().strftime(_DATE_FORMAT)}.")

    if not lines:
        return None
    return _wrap("\n\n".join(lines))


async def build_todos_reminder(session_id: str | None = None) -> str | None:
    """Render the session's todo board as a trailing reminder.

    Best-effort: returns None on any read error (no active env in some import
    paths, walker failure, etc.) or when the folder is empty.
    """
    try:
        todos = await read_todos(session_id)
    except Exception:
        return None
    rendered = render_todos(todos)
    if not rendered.strip():
        return None
    return _wrap(rendered)


def _append_reminder(llm_request: LlmRequest, text: str) -> None:
    llm_request.contents.append(
        types.Content(role="user", parts=[types.Part(text=text)])
    )


def make_reminder_injection_callback(
    *,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> Callable[[CallbackContext, LlmRequest], Awaitable[LlmResponse | None]]:
    """before_model_callback that appends <system-reminder> Content to the tail.

    Never short-circuits (always returns None). Volatile state, the rendered
    todo board, the near-budget warning, and the last-error nudge ride in the
    rolling message tail so the cached system prefix stays stable.
    """

    async def _callback(
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        state_obj = getattr(callback_context, "state", None)
        # ADK State is not a plain Mapping: dict(State) falls back to sequence
        # iteration and raises KeyError(0). Use its to_dict() snapshot.
        if hasattr(state_obj, "to_dict"):
            state = state_obj.to_dict()
        else:
            state = dict(state_obj or {})

        if state.get(_GUARDRAILS_HANDOFF_DELIVERED_STATE_KEY) or state.get(
            _BUDGET_HANDOFF_DELIVERED_STATE_KEY
        ):
            return None

        volatile = build_volatile_reminder(state=state)
        if volatile:
            _append_reminder(llm_request, volatile)

        todos = await build_todos_reminder(session_id_of(callback_context))
        if todos:
            _append_reminder(llm_request, todos)

        budget = build_budget_reminder(
            iteration=int(state.get("iteration") or 0),
            max_iterations=max_iterations,
        )
        if budget:
            _append_reminder(llm_request, budget)

        error = build_error_reminder(last_error=state.get("last_error"))
        if error:
            _append_reminder(llm_request, error)

        return None

    return _callback


reminder_injection_callback = make_reminder_injection_callback()


__all__ = [
    "REMINDER_CLOSE",
    "REMINDER_OPEN",
    "build_budget_reminder",
    "build_error_reminder",
    "build_todos_reminder",
    "build_volatile_reminder",
    "make_reminder_injection_callback",
    "reminder_injection_callback",
]
