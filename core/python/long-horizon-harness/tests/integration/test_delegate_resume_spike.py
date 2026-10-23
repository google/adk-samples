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

"""Spike: does ADK resumability repeat a prior side effect across a HITL pause?

The proposed delegate redesign rests on ONE assumption: if a child runner is
resumable, then pausing on a tool that needs approval and resuming after the
human confirms must NOT re-execute tool calls that already completed before the
pause. This drives a real Runner (resumable App + a scripted BaseLlm, no Vertex)
through pause → confirm → resume and asserts the prior side effect ran exactly
once.

If this fails, the "fail → approve → resume same session" design is unsound and
we fall back to the capability-grouped spawn gate.
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

# Module-level observable side effects, in execution order.
SIDE_EFFECTS: list[str] = []


def do_side_effect(tool_context: ToolContext) -> dict[str, Any]:
    """A normal, side-effecting tool that completes BEFORE the pause."""
    SIDE_EFFECTS.append("A")
    return {"status": "ok"}


def needs_approval(tool_context: ToolContext) -> dict[str, Any]:
    """A tool that pauses for HITL, then does its real work once confirmed."""
    tc = getattr(tool_context, "tool_confirmation", None)
    if tc is None:
        tool_context.request_confirmation(hint="approve?", payload={"x": 1})
        SIDE_EFFECTS.append("B:req")
        return {"status": "pending"}
    if getattr(tc, "confirmed", False):
        SIDE_EFFECTS.append("B:ok")
        return {"status": "done"}
    SIDE_EFFECTS.append("B:declined")
    return {"status": "declined"}


def _names(contents: list[types.Content] | None, attr: str) -> set[str]:
    out: set[str] = set()
    for content in contents or []:
        for part in content.parts or []:
            obj = getattr(part, attr, None)
            if obj is not None and getattr(obj, "name", None):
                out.add(obj.name)
    return out


def _fc_response(name: str) -> LlmResponse:
    fc = types.FunctionCall(name=name, args={})
    return LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part(function_call=fc)]
        )
    )


class _ScriptedLlm(BaseLlm):
    """Deterministic, stateless: decides the next step from the contents so far."""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        called = _names(llm_request.contents, "function_call")
        if "do_side_effect" not in called:
            yield _fc_response("do_side_effect")
        elif "needs_approval" not in called:
            yield _fc_response("needs_approval")
        else:
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="done")]
                )
            )


def _confirmation_call_id(events: list[Any]) -> str | None:
    for event in events:
        for fc in event.get_function_calls() or []:
            if fc.name == _CONFIRM_FN:
                return fc.id
    return None


@pytest.mark.asyncio
async def test_resume_does_not_repeat_prior_side_effect() -> None:
    SIDE_EFFECTS.clear()
    app = App(
        name="resume_spike",
        root_agent=LlmAgent(
            name="child",
            model=_ScriptedLlm(model="scripted"),
            tools=[do_side_effect, needs_approval],
        ),
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)
    await session_service.create_session(
        app_name="resume_spike", user_id="u", session_id="s"
    )

    # Run 1: do_side_effect completes, needs_approval pauses for confirmation.
    run1: list[Any] = []
    async for event in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(role="user", parts=[types.Part(text="go")]),
    ):
        run1.append(event)

    assert SIDE_EFFECTS == ["A", "B:req"], SIDE_EFFECTS
    conf_id = _confirmation_call_id(run1)
    assert conf_id is not None, (
        "expected a paused adk_request_confirmation call"
    )

    # Resume: human approves. ToolConfirmation fields go DIRECTLY in `response`
    # (the parser model_validates it; extra keys are forbidden).
    resume_msg = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=conf_id,
                    name=_CONFIRM_FN,
                    response={
                        "confirmed": True,
                        "hint": "approve?",
                        "payload": {"x": 1},
                    },
                )
            )
        ],
    )
    async for _ in runner.run_async(
        user_id="u", session_id="s", new_message=resume_msg
    ):
        pass

    # The confirmed tool ran after approval (proves resume actually did work)...
    assert "B:ok" in SIDE_EFFECTS, SIDE_EFFECTS
    # ...and the prior, already-completed side effect did NOT run a second time.
    assert SIDE_EFFECTS.count("A") == 1, SIDE_EFFECTS
