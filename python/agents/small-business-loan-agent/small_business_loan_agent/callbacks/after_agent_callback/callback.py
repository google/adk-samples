# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLM-as-Judge Gate for the Small Business Loan Agent.

Validates agent responses before they are shown to users by checking:
1. Trajectory correctness — Did the agent follow the expected tool sequence?
2. Grounding — Are all values traceable to agent outputs?
3. Response quality — Is the response complete and actionable?
"""

import os
import json

from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.genai import Client, types as genai_types
from google.genai.types import GenerateContentConfig
from small_business_loan_agent.callbacks.after_agent_callback.models import JudgeVerdict
from small_business_loan_agent.callbacks.after_agent_callback.prompt import JUDGE_PROMPT
from small_business_loan_agent.shared_libraries.firestore_utils.state_service import (
    ProcessStateService,
)
from small_business_loan_agent.shared_libraries.logging_config import get_logger

logger = get_logger(__name__)

JUDGE_MODEL = "gemini-3.1-pro-preview"

AGENT_OUTPUT_KEYS = [
    "DocumentExtractionAgent_output",
    "UnderwritingAgent_output",
    "PricingAgent_output",
    "LoanDecisionAgent_output",
]


def _parse_event_parts(event: Any) -> tuple[list[str], str | None, str | None]:
    """Parse a single event's parts to extract tool calls and text content."""
    content = getattr(event, "content", None)
    if not content:
        return [], None, None

    parts = getattr(content, "parts", None) or []
    role = getattr(content, "role", None)

    tool_names: list[str] = []
    user_text = None
    model_text = None

    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call:
            tool_name = getattr(function_call, "name", None)
            if tool_name:
                tool_names.append(tool_name)

        text = getattr(part, "text", None)
        if text:
            if role == "user":
                user_text = text
            elif role == "model":
                model_text = text

    return tool_names, user_text, model_text


def _extract_tool_sequence_and_messages(
    callback_context: CallbackContext,
) -> tuple[list[str], str, str]:
    """Extract tool call sequence, user message, and final model response from current invocation."""
    invocation_context = callback_context._invocation_context
    current_invocation_id = getattr(invocation_context, "invocation_id", None)

    session = invocation_context.session
    events = session.events if session and session.events else []

    tool_sequence: list[str] = []
    final_response = ""
    user_message = ""

    for event in events:
        event_invocation_id = getattr(event, "invocation_id", None)
        if current_invocation_id and event_invocation_id != current_invocation_id:
            continue

        tools, user_text, model_text = _parse_event_parts(event)
        tool_sequence.extend(tools)
        if user_text:
            user_message = user_text
        if model_text:
            final_response = model_text

    return tool_sequence, final_response, user_message


def _build_process_history(request_id: str | None, tool_sequence: list[str]) -> str:
    """Describe which workflow steps are already completed, across all turns, for this request.

    The judge only sees this turn's tool calls in `tool_sequence`, which makes a
    legitimate skip (a step completed in an earlier turn) indistinguishable from a
    genuine trajectory bug. This resolves the ambiguity from persisted process state:
    any ALL_STEPS entry marked completed/approved that ISN'T in this turn's
    tool_sequence must have been completed in a prior turn (if it had been completed
    THIS turn, its AgentTool call would appear in tool_sequence).
    """
    if not request_id:
        return "No request ID available — cannot look up process history."

    try:
        process_state = ProcessStateService().get_process_status(request_id)
    except Exception as e:
        return f"Process history unavailable (lookup error: {e})"

    if not process_state:
        return "No process history found for this request (first turn)."

    steps = process_state.get("steps", {})
    lines = []
    for step_name in ProcessStateService.ALL_STEPS:
        status = steps.get(step_name, {}).get("status", ProcessStateService.STATUS_NOT_STARTED)
        is_done = status in (ProcessStateService.STATUS_COMPLETED, ProcessStateService.STATUS_APPROVED)
        if is_done and step_name in tool_sequence:
            lines.append(f"- {step_name}: completed THIS turn")
        elif is_done:
            lines.append(f"- {step_name}: already completed (prior turn) — not called this turn, and that's expected")
        else:
            lines.append(f"- {step_name}: not completed (status={status})")

    return "\n".join(lines)


def _collect_agent_outputs(callback_context: CallbackContext) -> dict:
    """Collect agent outputs from session state for grounding checks."""
    agent_outputs = {}
    for key in AGENT_OUTPUT_KEYS:
        value = callback_context.state.get(key)
        if value:
            if hasattr(value, "model_dump"):
                agent_outputs[key] = value.model_dump()
            elif isinstance(value, dict):
                agent_outputs[key] = value
            else:
                agent_outputs[key] = str(value)
    return agent_outputs


async def llm_judge_gate(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """
    After-agent callback that validates the orchestrator's response before showing to user.

    Returns None to allow the response through, or Content with a blocked message.
    """
    request_id = callback_context.state.get("loan_request_id")

    try:
        tool_sequence, final_response, user_message = _extract_tool_sequence_and_messages(callback_context)
        agent_outputs = _collect_agent_outputs(callback_context)
        process_history = _build_process_history(request_id, tool_sequence)

        logger.info(f"LLM Judge evaluating response for {request_id}")
        logger.info(f"Tool sequence: {' -> '.join(tool_sequence)}")

        judge_input = JUDGE_PROMPT.format(
            agent_outputs=json.dumps(agent_outputs, indent=2, default=str) if agent_outputs else "No agent outputs",
            tool_sequence=" -> ".join(tool_sequence) if tool_sequence else "No tools called",
            process_history=process_history,
            user_message=user_message or "No user message",
            final_response=final_response or "No response",
        )

        client = Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="global")
        judge_response = await client.aio.models.generate_content(
            model=JUDGE_MODEL,
            contents=judge_input,
            config=GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JudgeVerdict,
            ),
        )

        verdict_text = judge_response.text or ""
        verdict = JudgeVerdict.model_validate_json(verdict_text)

        log_msg = (
            f"LLM Judge {'APPROVED' if verdict.is_valid else 'BLOCKED'} response for {request_id} "
            f"(trajectory={verdict.trajectory_correct}, grounded={verdict.grounded_in_context})"
        )
        if verdict.is_valid:
            logger.info(log_msg)
        else:
            logger.warning(log_msg)
            logger.warning(f"Reasoning: {verdict.reasoning}")

        callback_context.state["_llm_judge_audit"] = {
            "loan_request_id": request_id,
            "verdict": verdict.model_dump(),
            "tool_sequence": tool_sequence,
        }

        if not verdict.is_valid:
            return genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        text=(
                            "I apologize, but I need to verify some information before providing a response. "
                            "This is to ensure accuracy in your loan processing request.\n\n"
                            f"Reference ID: {request_id or 'N/A'}\n\n"
                            "Please try your request again, or contact support if this issue persists."
                        )
                    )
                ],
            )

        return None

    except Exception as e:
        logger.error(f"LLM Judge error: {e}", exc_info=True)
        return genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    text=(
                        "I apologize, but I encountered an issue while validating the response. "
                        f"Please try your request again.\n\nReference: {request_id or 'N/A'}"
                    )
                )
            ],
        )
