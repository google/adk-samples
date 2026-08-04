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

"""Deterministic tests for the no-LLM tool-output pruning transform.

We assert only on the event-list mutation (which bodies got zeroed, which were
protected) and the reclaimed-token bookkeeping — never on any model output.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.adk.events.event import Event
from google.genai.types import Content, FunctionResponse, Part

from horizon.context.tool_output_pruning import (
    PRUNE_MARKER,
    prune_tool_outputs,
    prune_tool_outputs_callback,
)


def _tool_event(name: str, body: str, *, ts: float) -> Event:
    part = Part(
        function_response=FunctionResponse(
            id=f"{name}-{ts}", name=name, response={"output": body}
        )
    )
    return Event(
        author="user",
        content=Content(role="user", parts=[part]),
        invocation_id=f"inv-{ts}",
        timestamp=ts,
    )


def _user_event(text: str, *, ts: float) -> Event:
    return Event(
        author="user",
        content=Content(role="user", parts=[Part(text=text)]),
        invocation_id=f"user-{ts}",
        timestamp=ts,
    )


def _big(n: int) -> str:
    return "x" * n


def _resp_body(event: Event) -> object:
    return event.content.parts[0].function_response.response


class TestPruneTransform:
    def test_old_large_tool_output_is_zeroed(self) -> None:
        # Three turns; protect the most recent 1 turn. Oldest big read_file
        # output should be pruned.
        events = [
            _tool_event("read_file", _big(80_000), ts=1.0),  # old, big -> prune
            _user_event("next", ts=2.0),
            _tool_event(
                "read_file", _big(80_000), ts=3.0
            ),  # recent -> protected
            _user_event("now", ts=4.0),
        ]
        result = prune_tool_outputs(
            events, protect_recent_turns=1, protect_token_budget=0
        )
        assert result.reclaimed_tokens > 0
        assert result.pruned_count == 1
        assert _resp_body(events[0]) == {"pruned": PRUNE_MARKER}
        # Recent one untouched.
        assert _resp_body(events[2]) == {"output": _big(80_000)}

    def test_skill_outputs_are_never_pruned(self) -> None:
        events = [
            _tool_event("load_skill", _big(80_000), ts=1.0),
            _user_event("turn", ts=2.0),
            _user_event("turn2", ts=3.0),
        ]
        result = prune_tool_outputs(
            events, protect_recent_turns=0, protect_token_budget=0
        )
        assert result.pruned_count == 0
        assert _resp_body(events[0]) == {"output": _big(80_000)}

    def test_small_outputs_below_floor_not_pruned(self) -> None:
        events = [
            _tool_event("read_file", "tiny", ts=1.0),
            _user_event("turn", ts=2.0),
            _user_event("turn2", ts=3.0),
        ]
        result = prune_tool_outputs(
            events,
            protect_recent_turns=0,
            protect_token_budget=0,
            min_part_tokens=1_000,
        )
        assert result.pruned_count == 0

    def test_no_change_when_reclaim_below_minimum(self) -> None:
        # One prunable part just over min_part_tokens but under min_reclaim:
        # nothing is mutated and reclaimed is reported as 0.
        events = [
            _tool_event("read_file", _big(5_000), ts=1.0),  # ~1250 tokens
            _user_event("turn", ts=2.0),
            _user_event("turn2", ts=3.0),
        ]
        result = prune_tool_outputs(
            events,
            protect_recent_turns=0,
            protect_token_budget=0,
            min_part_tokens=100,
            min_reclaim_tokens=100_000,
        )
        assert result.pruned_count == 0
        assert result.reclaimed_tokens == 0
        assert _resp_body(events[0]) == {"output": _big(5_000)}

    def test_already_pruned_part_is_skipped(self) -> None:
        events = [
            _tool_event("read_file", _big(80_000), ts=1.0),
            _user_event("turn", ts=2.0),
            _user_event("turn2", ts=3.0),
        ]
        prune_tool_outputs(
            events, protect_recent_turns=0, protect_token_budget=0
        )
        # Second pass: nothing left to reclaim.
        result = prune_tool_outputs(
            events, protect_recent_turns=0, protect_token_budget=0
        )
        assert result.pruned_count == 0
        assert result.reclaimed_tokens == 0

    def test_protect_token_budget_keeps_recent_bodies(self) -> None:
        # Large budget keeps even old bodies because their cumulative estimate
        # stays within the protected budget.
        events = [
            _tool_event("read_file", _big(40_000), ts=1.0),
            _user_event("turn", ts=2.0),
            _tool_event("read_file", _big(40_000), ts=3.0),
            _user_event("turn2", ts=4.0),
        ]
        result = prune_tool_outputs(
            events,
            protect_recent_turns=0,
            protect_token_budget=1_000_000,
        )
        assert result.pruned_count == 0


def _prunable_session_events() -> list[Event]:
    # The callback runs with production defaults (protect 3 recent turns, 40k
    # token budget). To actually prune, the oldest tool body must sit beyond
    # the recent-turn window with enough newer tool volume ahead of it to
    # exhaust the protected budget.
    return [
        _tool_event("read_file", _big(240_000), ts=1.0),  # old target -> prune
        _user_event("a", ts=2.0),
        _tool_event("read_file", _big(60_000), ts=3.0),
        _user_event("b", ts=4.0),
        _tool_event("read_file", _big(60_000), ts=5.0),
        _user_event("c", ts=6.0),
        _tool_event("read_file", _big(60_000), ts=7.0),
        _user_event("d", ts=8.0),
    ]


@pytest.mark.asyncio
class TestPruneCallback:
    async def test_mutates_session_events(self) -> None:
        events = _prunable_session_events()
        session = SimpleNamespace(events=events)
        ctx = SimpleNamespace(
            _invocation_context=SimpleNamespace(session=session)
        )
        await prune_tool_outputs_callback(
            callback_context=ctx, llm_request=SimpleNamespace()
        )
        assert events[0].content.parts[0].function_response.response == {
            "pruned": PRUNE_MARKER
        }

    async def test_disabled_via_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LHA_PRUNE_TOOL_OUTPUTS", "0")
        events = _prunable_session_events()
        session = SimpleNamespace(events=events)
        ctx = SimpleNamespace(
            _invocation_context=SimpleNamespace(session=session)
        )
        await prune_tool_outputs_callback(
            callback_context=ctx, llm_request=SimpleNamespace()
        )
        assert events[0].content.parts[0].function_response.response == {
            "output": _big(240_000)
        }

    async def test_missing_context_is_noop(self) -> None:
        ctx = SimpleNamespace(_invocation_context=None)
        await prune_tool_outputs_callback(
            callback_context=ctx, llm_request=SimpleNamespace()
        )
