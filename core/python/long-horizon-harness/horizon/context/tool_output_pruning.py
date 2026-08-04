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

"""Deterministic, no-LLM pruning of stale tool-result bodies.

Compaction with an LLM is the heavy hammer. Before paying for it, we can
reclaim a lot of context for free by zeroing the bodies of *old, large* tool
results the model is unlikely to re-read (file dumps, search results, terminal
output). We never touch skill-tool outputs (they carry durable instructions),
never touch the most recent turns, and only mutate if the reclaim is worth it.

Walks newest->oldest so "recent" protection is a simple turn/token countdown.
Mutates the passed event list in place (ADK reads ``session.events`` directly
when estimating prompt tokens), and reports what it reclaimed so the caller can
log and decide.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from google.adk.events.event import Event

logger = logging.getLogger(__name__)

PRUNE_MARKER = "[output pruned to reclaim context — re-run the tool if needed]"
PRUNE_ENABLED_ENV = "LHA_PRUNE_TOOL_OUTPUTS"

# Tool names whose outputs are never pruned (substring match, case-insensitive).
PROTECTED_TOOL_SUBSTRINGS = ("skill",)

DEFAULT_PROTECT_RECENT_TURNS = 3
DEFAULT_PROTECT_TOKEN_BUDGET = 40_000
DEFAULT_MIN_PART_TOKENS = 2_000
DEFAULT_MIN_RECLAIM_TOKENS = 20_000


@dataclass
class PruneResult:
    pruned_count: int
    reclaimed_tokens: int


def _estimate_tokens(value: object) -> int:
    return len(str(value)) // 4


def _is_protected_tool(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return any(sub in lowered for sub in PROTECTED_TOOL_SUBSTRINGS)


def _is_user_turn(event: Event) -> bool:
    content = event.content
    if not content or not content.parts:
        return False
    return any(p.text for p in content.parts) and not any(
        p.function_response or p.function_call for p in content.parts
    )


def _already_pruned(response: object) -> bool:
    return isinstance(response, dict) and response.get("pruned") == PRUNE_MARKER


def prune_tool_outputs(
    events: list[Event],
    *,
    protect_recent_turns: int = DEFAULT_PROTECT_RECENT_TURNS,
    protect_token_budget: int = DEFAULT_PROTECT_TOKEN_BUDGET,
    min_part_tokens: int = DEFAULT_MIN_PART_TOKENS,
    min_reclaim_tokens: int = DEFAULT_MIN_RECLAIM_TOKENS,
) -> PruneResult:
    """Zero large, stale tool-result bodies in place.

    Returns the count/tokens it *would* reclaim; mutates only if the total
    reclaim meets ``min_reclaim_tokens`` (so we never thrash the history for a
    trivial gain).
    """
    candidates: list[tuple[Event, int, int]] = []  # (event, part_index, tokens)
    turns_seen = 0
    protected_tokens = 0
    for event in reversed(events):
        if _is_user_turn(event):
            turns_seen += 1
        content = event.content
        if not content or not content.parts:
            continue
        for idx, part in enumerate(content.parts):
            fr = part.function_response
            if fr is None:
                continue
            if _already_pruned(fr.response):
                continue
            tokens = _estimate_tokens(fr.response)
            if turns_seen <= protect_recent_turns:
                protected_tokens += tokens
                continue
            if protected_tokens < protect_token_budget:
                protected_tokens += tokens
                continue
            if _is_protected_tool(fr.name):
                continue
            if tokens < min_part_tokens:
                continue
            candidates.append((event, idx, tokens))

    reclaimable = sum(t for _, _, t in candidates)
    if reclaimable < min_reclaim_tokens:
        return PruneResult(pruned_count=0, reclaimed_tokens=0)

    for event, idx, _ in candidates:
        part = event.content.parts[idx]
        fr = part.function_response
        fr.response = {"pruned": PRUNE_MARKER}

    return PruneResult(
        pruned_count=len(candidates), reclaimed_tokens=reclaimable
    )


async def prune_tool_outputs_callback(
    *, callback_context: Any, llm_request: Any
) -> None:
    if os.environ.get(PRUNE_ENABLED_ENV, "1") == "0":
        return None
    inv = getattr(callback_context, "_invocation_context", None)
    session = getattr(inv, "session", None)
    events = getattr(session, "events", None)
    if not events:
        return None
    result = prune_tool_outputs(events)
    if result.pruned_count:
        logger.info(
            "tool_output_pruning: pruned %d part(s), ~%d tokens reclaimed",
            result.pruned_count,
            result.reclaimed_tokens,
        )
    return None
