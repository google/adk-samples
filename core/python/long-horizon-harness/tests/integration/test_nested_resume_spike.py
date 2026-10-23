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

"""Spike #2: nested two-runner HITL handoff (the mechanism the feature rests on).

Spike #1 proved a SINGLE resumable runner doesn't repeat prior side effects on
resume. The real `delegate` has TWO runners: the parent turn and an isolated
child. This proves the full handoff:

  child tool needs approval -> child runner PAUSES
  -> delegate() detects the pause and re-raises the confirmation on the PARENT
     runner (which surfaces to the user)
  -> parent turn PAUSES
  -> human approves -> parent RESUMES -> delegate() re-runs from the top, but
     REATTACHES to the persisted child session and RESUMES it (not restart)
  -> child runner resumes, runs only the confirmed tool

The invariant: the child's prior side effect runs EXACTLY ONCE, even though
delegate() itself is re-invoked from scratch on parent resume. If this holds,
"fail -> approve -> resume same session" is sound for blocking delegate.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

_CONFIRM_FN = "adk_request_confirmation"

# Observable child side effects, in execution order.
SIDE_EFFECTS: list[str] = []

# Child session state, keyed by the PARENT delegate call id. In production this
# would be durable storage; the spike only needs it to survive the parent's
# in-process pause/resume.
CHILD_STATE: dict[str, dict[str, str]] = {}

# A single child session service instance so the child session persists across
# the parent's pause/resume (two separate parent runner invocations).
CHILD_SESSIONS = InMemorySessionService()
_CHILD_USER = "child_user"


# ---- child ----------------------------------------------------------------


def child_side_effect(tool_context: ToolContext) -> dict[str, Any]:
    SIDE_EFFECTS.append("child:A")
    return {"status": "ok"}


def child_needs_approval(tool_context: ToolContext) -> dict[str, Any]:
    tc = getattr(tool_context, "tool_confirmation", None)
    if tc is None:
        tool_context.request_confirmation(
            hint="child wants to run X", payload={"op": "X"}
        )
        SIDE_EFFECTS.append("child:B:req")
        return {"status": "pending"}
    if getattr(tc, "confirmed", False):
        SIDE_EFFECTS.append("child:B:ok")
        return {"status": "done"}
    SIDE_EFFECTS.append("child:B:declined")
    return {"status": "declined"}


# ---- deterministic, stateless scripted models -----------------------------


def _called(contents: list[types.Content] | None) -> set[str]:
    out: set[str] = set()
    for content in contents or []:
        for part in content.parts or []:
            fc = getattr(part, "function_call", None)
            if fc is not None and fc.name:
                out.add(fc.name)
    return out


def _fc(name: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(function_call=types.FunctionCall(name=name, args={}))
            ],
        )
    )


def _text(s: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=s)])
    )


class _ChildLlm(BaseLlm):
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        called = _called(llm_request.contents)
        if "child_side_effect" not in called:
            yield _fc("child_side_effect")
        elif "child_needs_approval" not in called:
            yield _fc("child_needs_approval")
        else:
            yield _text("child done")


class _ParentLlm(BaseLlm):
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        called = _called(llm_request.contents)
        if "delegate" not in called:
            yield _fc("delegate")
        else:
            yield _text("parent done")


_CHILD_APP = App(
    name="child",
    root_agent=LlmAgent(
        name="child",
        model=_ChildLlm(model="child"),
        tools=[child_side_effect, child_needs_approval],
    ),
    resumability_config=ResumabilityConfig(is_resumable=True),
)


# ---- helpers over emitted events ------------------------------------------


def _confirmation_call(events: list[Any]) -> types.FunctionCall | None:
    for event in events:
        for fc in event.get_function_calls() or []:
            if fc.name == _CONFIRM_FN:
                return fc
    return None


# ---- the delegate-like tool that drives the child runner ------------------


async def delegate(tool_context: ToolContext) -> dict[str, Any]:
    """Blocking delegate: runs a child runner, bubbles child HITL to the parent."""
    parent_fc_id = tool_context.function_call_id
    child_runner = Runner(app=_CHILD_APP, session_service=CHILD_SESSIONS)
    confirmation = getattr(tool_context, "tool_confirmation", None)

    if confirmation is None:
        session = await CHILD_SESSIONS.create_session(
            app_name="child", user_id=_CHILD_USER
        )
        CHILD_STATE[parent_fc_id] = {"sid": session.id}
        events = [
            e
            async for e in child_runner.run_async(
                user_id=_CHILD_USER,
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="go")]
                ),
            )
        ]
        pending = _confirmation_call(events)
        if pending is not None:
            CHILD_STATE[parent_fc_id]["child_conf_id"] = pending.id
            tc = (pending.args or {}).get("toolConfirmation", {})
            tool_context.request_confirmation(
                hint=f"[child] {tc.get('hint', 'approve?')}",
                payload=tc.get("payload"),
            )
            return {"status": "pending_child"}
        return {"status": "done", "summary": "child finished without approval"}

    # Parent resumed: reattach + resume the child in place.
    state = CHILD_STATE[parent_fc_id]
    resume_msg = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=state["child_conf_id"],
                    name=_CONFIRM_FN,
                    response={
                        "confirmed": bool(
                            getattr(confirmation, "confirmed", False)
                        )
                    },
                )
            )
        ],
    )
    async for _ in child_runner.run_async(
        user_id=_CHILD_USER, session_id=state["sid"], new_message=resume_msg
    ):
        pass
    return {"status": "done", "summary": "child resumed and finished"}


@pytest.mark.asyncio
async def test_nested_handoff_runs_child_side_effect_exactly_once() -> None:
    SIDE_EFFECTS.clear()
    CHILD_STATE.clear()

    parent_app = App(
        name="parent",
        root_agent=LlmAgent(
            name="parent", model=_ParentLlm(model="parent"), tools=[delegate]
        ),
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    parent_sessions = InMemorySessionService()
    parent_runner = Runner(app=parent_app, session_service=parent_sessions)
    await parent_sessions.create_session(
        app_name="parent", user_id="u", session_id="s"
    )

    # Parent turn 1: delegate runs the child; child pauses; delegate bubbles the
    # confirmation up; the PARENT turn pauses.
    run1: list[Any] = []
    async for event in parent_runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(
            role="user", parts=[types.Part(text="do it")]
        ),
    ):
        run1.append(event)

    assert SIDE_EFFECTS == ["child:A", "child:B:req"], SIDE_EFFECTS
    parent_conf = _confirmation_call(run1)
    assert parent_conf is not None, (
        "parent turn should have paused for confirmation"
    )

    # Human approves on the PARENT turn -> parent resumes -> delegate re-runs from
    # the top, reattaches to the child session, and resumes the child in place.
    resume_msg = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=parent_conf.id,
                    name=_CONFIRM_FN,
                    response={"confirmed": True},
                )
            )
        ],
    )
    async for _ in parent_runner.run_async(
        user_id="u", session_id="s", new_message=resume_msg
    ):
        pass

    # The confirmed child tool ran after approval...
    assert "child:B:ok" in SIDE_EFFECTS, SIDE_EFFECTS
    # ...and the child's prior side effect ran EXACTLY ONCE, despite delegate()
    # itself being re-invoked from scratch on parent resume.
    assert SIDE_EFFECTS.count("child:A") == 1, SIDE_EFFECTS
