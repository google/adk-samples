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

"""Tests for the report_to_maintainers tool — agent self-report, HITL-gated.

First call → ``request_user_confirmation`` + pending envelope; nothing emitted.
Resume confirmed → one agent-origin record through the feedback sink. Resume
declined → nothing emitted. A repeat of the same (category, summary) in one
session short-circuits without a prompt. Conversation context is attached only
when the agent opted in AND the user confirmed.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import horizon.tools.report_feedback as rf
from horizon.tools.report_feedback import report_to_maintainers


def _event(author: str, *texts: str) -> SimpleNamespace:
    parts = [SimpleNamespace(text=t) for t in texts]
    return SimpleNamespace(author=author, content=SimpleNamespace(parts=parts))


def _fake_context(
    *,
    tool_confirmation: SimpleNamespace | None = None,
    state: dict | None = None,
    events: list | None = None,
) -> SimpleNamespace:
    shared_state = state if state is not None else {}
    inv = SimpleNamespace(
        session=SimpleNamespace(
            id="sess-1", events=events or [], state=shared_state
        ),
        user_id="user-1",
    )
    return SimpleNamespace(
        state=shared_state,
        tool_confirmation=tool_confirmation,
        actions=SimpleNamespace(skip_summarization=False),
        request_confirmation=MagicMock(),
        _invocation_context=inv,
    )


@pytest.fixture(autouse=True)
def _no_real_emit(monkeypatch):
    emit = MagicMock()
    monkeypatch.setattr(rf, "emit_feedback_record", emit)
    return emit


def test_first_call_requests_confirmation_and_pends(_no_real_emit):
    ctx = _fake_context()
    result = report_to_maintainers(
        category="bug",
        summary="terminal crashes on long output",
        details="ran X, got Y, expected Z",
        tool_context=ctx,
    )

    assert result["status"] == "awaiting_user_response"
    ctx.request_confirmation.assert_called_once()
    assert ctx.actions.skip_summarization is True
    _no_real_emit.assert_not_called()


def test_declined_emits_nothing(_no_real_emit):
    ctx = _fake_context(
        tool_confirmation=SimpleNamespace(confirmed=False, payload=None)
    )
    result = report_to_maintainers(
        category="bug",
        summary="x",
        details="y",
        tool_context=ctx,
    )

    assert result["success"] is False
    assert result["status"] == "declined"
    _no_real_emit.assert_not_called()


def test_confirmed_emits_agent_record(_no_real_emit):
    ctx = _fake_context(
        tool_confirmation=SimpleNamespace(confirmed=True, payload=None)
    )
    result = report_to_maintainers(
        category="guardrail_misfire",
        summary="egress blocked a benign GET",
        details="curl to example.com was blocked",
        tool_context=ctx,
    )

    assert result["status"] == "filed"
    _no_real_emit.assert_called_once()
    record = _no_real_emit.call_args.args[0]
    assert record["origin"] == "agent"
    assert record["category"] == "guardrail_misfire"
    assert record["log_type"] == "feedback"
    assert record["session_id"] == "sess-1"
    assert record["user_id"] == "user-1"
    assert "egress blocked a benign GET" in record["text"]
    assert "context_summary" not in record


def test_context_attached_only_when_opted_in_and_confirmed(_no_real_emit):
    ctx = _fake_context(
        tool_confirmation=SimpleNamespace(confirmed=True, payload=None),
        events=[_event("user", "this is the bug"), _event("root_agent", "ack")],
    )
    report_to_maintainers(
        category="bug",
        summary="s",
        details="d",
        include_context=True,
        tool_context=ctx,
    )

    record = _no_real_emit.call_args.args[0]
    assert "context_summary" in record
    assert "this is the bug" in record["context_summary"]


def test_dedupe_short_circuits(_no_real_emit):
    state = {"reported_signatures": ["bug|already filed"]}
    ctx = _fake_context(state=state)
    result = report_to_maintainers(
        category="bug",
        summary="Already filed",
        details="whatever",
        tool_context=ctx,
    )

    assert result["status"] == "already_reported"
    ctx.request_confirmation.assert_not_called()
    _no_real_emit.assert_not_called()


def test_signature_recorded_after_filing(_no_real_emit):
    state: dict = {}
    ctx = _fake_context(
        tool_confirmation=SimpleNamespace(confirmed=True, payload=None),
        state=state,
    )
    report_to_maintainers(
        category="bug",
        summary="New bug",
        details="d",
        tool_context=ctx,
    )

    assert "bug|new bug" in state["reported_signatures"]


def test_invalid_category_rejected(_no_real_emit):
    ctx = _fake_context()
    result = report_to_maintainers(
        category="meh",  # type: ignore[arg-type]
        summary="s",
        details="d",
        tool_context=ctx,
    )

    assert result["success"] is False
    ctx.request_confirmation.assert_not_called()


def test_blank_summary_rejected(_no_real_emit):
    ctx = _fake_context()
    result = report_to_maintainers(
        category="bug",
        summary="   ",
        details="d",
        tool_context=ctx,
    )

    assert result["success"] is False
    ctx.request_confirmation.assert_not_called()


def test_blank_details_rejected(_no_real_emit):
    ctx = _fake_context()
    result = report_to_maintainers(
        category="bug",
        summary="s",
        details="",
        tool_context=ctx,
    )

    assert result["success"] is False
    ctx.request_confirmation.assert_not_called()


def test_tool_context_is_keyword_only():
    sig = inspect.signature(report_to_maintainers)
    assert sig.parameters["tool_context"].kind == inspect.Parameter.KEYWORD_ONLY


def test_registered_on_root_agent():
    from horizon.agent import root_agent

    assert report_to_maintainers in root_agent.tools
