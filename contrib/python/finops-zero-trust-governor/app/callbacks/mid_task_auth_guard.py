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

"""Unified 5-Gate Enterprise Governance Harness.

Solves 5 critical ADK production bottlenecks without bloating prompts:
  1. Gate 1 — Token & Retry Circuit Breaker (before_model_callback)
  2. Gate 2 — Mid-Task OAuth/IAM Revocation (before_tool_callback, solves Issue #2769)
  3. Gate 3 — Crash-Resumable Tool Idempotency (before_tool_callback)
  4. Gate 4 — Concurrent Workload Mutex Lock (before_tool_callback)
  5. Gate 5 — Blast-Radius HITL Escalation Gate (before_tool_callback)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

PROTECTED_MUTATION_TOOLS = {
    "apply_model_routing_downgrade",
    "update_cloud_run_scaling_limits",
}


def enforce_token_and_retry_circuit_breaker(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Gate 1 (before_model_callback): Prevents runaway LLM retry loops and token burn spikes."""
    max_turns = int(os.environ.get("FINOPS_MAX_LLM_TURNS_PER_RUN", "12"))
    current_turns = (
        int(callback_context.state.get("invocation_llm_turns", 0)) + 1
    )
    callback_context.state["invocation_llm_turns"] = current_turns

    if current_turns > max_turns:
        audit_log = callback_context.state.get("security_audit_log", [])
        audit_log.append(
            {
                "event_type": "CIRCUIT_BREAKER_TRIPPED",
                "turn_count": current_turns,
                "max_allowed_turns": max_turns,
                "timestamp_epoch": int(time.time()),
            }
        )
        callback_context.state["security_audit_log"] = audit_log
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            f"CIRCUIT_BREAKER_TRIPPED: Invocation exceeded the maximum allowed "
                            f"LLM turn budget ({max_turns} turns). Halting execution to prevent "
                            f"runaway Vertex AI token consumption."
                        )
                    )
                ],
            ),
        )
    return None


def verify_token_not_revoked(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
    """ADK before_tool_callback enforcing Gates 2, 3, 4, and 5 before state-mutating FinOps tools.

    Gate 2: Mid-Task OAuth/IAM Revocation Check (Solves Issue #2769)
    Gate 3: Crash-Resumable Tool Idempotency Guard (ResumabilityConfig at-least-once protection)
    Gate 4: Concurrent Workload Mutex Lock
    Gate 5: Blast-Radius Human-in-the-Loop (HITL) Escalation Gate
    """
    enforce = (
        os.environ.get("ZERO_TRUST_ENFORCE_REVOCATION", "TRUE").upper()
        == "TRUE"
    )
    if not enforce or tool.name not in PROTECTED_MUTATION_TOOLS:
        return None

    audit_log = tool_context.state.get("security_audit_log", [])
    workload_label = str(args.get("workload_label", "default"))
    current_invocation_id = str(
        getattr(tool_context, "invocation_id", None)
        or tool_context.state.get("invocation_id", "inv-default")
    )

    # -------------------------------------------------------------------------
    # GATE 2: Mid-Task OAuth / IAM Session Revocation Check (Issue #2769)
    # -------------------------------------------------------------------------
    session_revoked = bool(tool_context.state.get("auth_token_revoked", False))
    access_token = tool_context.state.get("oauth_access_token")
    is_test = os.environ.get("INTEGRATION_TEST", "TRUE").upper() == "TRUE"

    if not is_test and access_token and not session_revoked:
        tokeninfo_url = os.environ.get(
            "ZERO_TRUST_TOKENINFO_URL",
            "https://oauth2.googleapis.com/tokeninfo",
        )
        req_url = f"{tokeninfo_url}?access_token={access_token}"
        try:
            with urllib.request.urlopen(req_url, timeout=3.0) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
                if int(payload.get("expires_in", 0)) <= 0:
                    session_revoked = True
        except (urllib.error.URLError, ValueError, KeyError):
            session_revoked = True

    if session_revoked:
        event = {
            "event_type": "MID_TASK_AUTH_REVOCATION_BLOCK",
            "blocked_tool": tool.name,
            "workload_label": workload_label,
            "attempted_args": args,
            "timestamp_epoch": int(time.time()),
            "reason": (
                "Zero-Trust Guardrail: User OAuth/IAM session was revoked mid-trajectory "
                "after workflow initialization. Destructive/write tool execution aborted."
            ),
        }
        audit_log.append(event)
        tool_context.state["security_audit_log"] = audit_log
        return {
            "status": "BLOCKED_BY_ZERO_TRUST_GUARDRAIL",
            "error_code": "AUTH_TOKEN_REVOKED_MID_TASK",
            "blocked_tool": tool.name,
            "message": event["reason"],
        }

    # -------------------------------------------------------------------------
    # GATE 3: Crash-Resumable Tool Idempotency Guard (ResumabilityConfig Replay)
    # -------------------------------------------------------------------------
    idempotency_key = f"{tool.name}:{workload_label}"
    completed_tools = tool_context.state.get("completed_idempotent_tools", {})
    if idempotency_key in completed_tools:
        cached_result = dict(completed_tools[idempotency_key])
        cached_result["resumed_from_checkpoint"] = True
        cached_result["idempotency_key"] = idempotency_key
        audit_log.append(
            {
                "event_type": "IDEMPOTENT_REPLAY_SHORT_CIRCUIT",
                "tool": tool.name,
                "idempotency_key": idempotency_key,
                "timestamp_epoch": int(time.time()),
            }
        )
        tool_context.state["security_audit_log"] = audit_log
        return cached_result

    # -------------------------------------------------------------------------
    # GATE 4: Concurrent Workload Mutex Lock
    # -------------------------------------------------------------------------
    active_locks = tool_context.state.get("active_workload_locks", {})
    lock_owner = active_locks.get(workload_label)
    if lock_owner and lock_owner != current_invocation_id:
        event = {
            "event_type": "CONCURRENT_MUTATION_LOCK_BLOCK",
            "blocked_tool": tool.name,
            "workload_label": workload_label,
            "locked_by_invocation": lock_owner,
            "timestamp_epoch": int(time.time()),
        }
        audit_log.append(event)
        tool_context.state["security_audit_log"] = audit_log
        return {
            "status": "BLOCKED_BY_CONCURRENCY_GUARDRAIL",
            "error_code": "CONCURRENT_MUTATION_LOCK_CONFLICT",
            "blocked_tool": tool.name,
            "message": (
                f"Workload '{workload_label}' is currently locked for mutation by "
                f"concurrent invocation '{lock_owner}'."
            ),
        }
    active_locks[workload_label] = current_invocation_id
    tool_context.state["active_workload_locks"] = active_locks

    # -------------------------------------------------------------------------
    # GATE 5: Blast-Radius Human-in-the-Loop (HITL) Threshold Gate
    # -------------------------------------------------------------------------
    if (
        tool.name == "update_cloud_run_scaling_limits"
        and int(args.get("new_min_instances", 1)) == 0
        and not bool(tool_context.state.get("human_hitl_approved", False))
    ):
        event = {
            "event_type": "BLAST_RADIUS_HITL_ESCALATION",
            "blocked_tool": tool.name,
            "workload_label": workload_label,
            "attempted_args": args,
            "timestamp_epoch": int(time.time()),
        }
        audit_log.append(event)
        tool_context.state["security_audit_log"] = audit_log
        return {
            "status": "PAUSED_FOR_HITL_APPROVAL",
            "error_code": "BLAST_RADIUS_EXCEEDS_AUTONOMOUS_LIMIT",
            "blocked_tool": tool.name,
            "message": (
                "Scaling Cloud Run min_instances to 0 on an active production workload "
                "exceeds autonomous blast-radius limits. Set session state "
                "`human_hitl_approved=True` and resume the invocation to proceed."
            ),
        }

    # All 5 gates passed
    audit_log.append(
        {
            "event_type": "MID_TASK_AUTH_VERIFIED",
            "verified_tool": tool.name,
            "workload_label": workload_label,
            "timestamp_epoch": int(time.time()),
        }
    )
    tool_context.state["security_audit_log"] = audit_log
    return None
