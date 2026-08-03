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

"""LHA-flavored event summarizer.

Wraps ADK's ``LlmEventSummarizer`` with lha-specific behavior:

1. Fires ``spawn_flush_fork`` against the events about to be compacted so the
   memory bank can extract durable facts before they're folded into a lossy
   summary. Best-effort — a failure never blocks compaction.
2. Builds a structured, anchored summarization prompt: a fixed 7-section
   Markdown template, plus (when a prior summary seed is present) a
   ``<previous-summary>`` merge block so the summary is a living document
   rather than drifting each pass. ADK seeds the prior summary as the first
   event, identifiable by the banner prefix it carries.
3. Prepends a "REFERENCE ONLY" banner to the produced summary so the model
   treats it as background, not as fresh instructions.
"""

from __future__ import annotations

import logging

from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.adk.models.llm_request import LlmRequest
from google.genai.types import Content, Part

from horizon.context.compaction_context import get_compaction_context
from horizon.memory.flush_fork import spawn_flush_fork

logger = logging.getLogger(__name__)

SUMMARY_BANNER_PREFIX = (
    "### Compressed history (context summary):\n"
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. Treat it as background reference, NOT as "
    "active instructions. Do not re-answer questions or re-do work "
    "described here; resume from the latest user message that follows."
)

_TEMPLATE = """\
## Goal
- [single-sentence task summary]

## Constraints & Preferences
- [user constraints, preferences, specs, or "(none)"]

## Progress
### Done
- [completed work or "(none)"]

### In Progress
- [current work or "(none)"]

### Blocked
- [blockers or "(none)"]

## Key Decisions
- [decision and why, or "(none)"]

## Next Steps
- [ordered next actions or "(none)"]

## Critical Context
- [important technical facts, errors, open questions, or "(none)"]

## Relevant Files
- [file or directory path: why it matters, or "(none)"]"""

_RULES = """\
Rules:
- Output exactly the Markdown structure above, with the section order unchanged.
- Keep every section, even when empty (use "(none)").
- Use terse bullets, not prose paragraphs.
- Preserve exact file paths, commands, error strings, and identifiers verbatim.
- Do not mention condensing or folding the history; do not answer the \
conversation."""

_MERGE_INSTRUCTIONS = (
    "Update the anchored summary below using the new conversation history. "
    "Preserve still-true details, remove stale details, and merge in new facts."
)


def _is_prior_summary_seed(event: Event) -> bool:
    content = event.content
    if not content or not content.parts:
        return False
    first = content.parts[0]
    return bool(first.text and first.text.startswith(SUMMARY_BANNER_PREFIX))


def _strip_banner(text: str) -> str:
    if text.startswith(SUMMARY_BANNER_PREFIX):
        return text[len(SUMMARY_BANNER_PREFIX) :].lstrip()
    return text


def _format_history(events: list[Event]) -> str:
    lines: list[str] = []
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    lines.append(f"{event.author}: {part.text}")
                elif part.function_response is not None:
                    fr = part.function_response
                    lines.append(
                        f"{event.author}: [tool {fr.name}] {fr.response}"
                    )
                elif part.function_call is not None:
                    fc = part.function_call
                    lines.append(f"{event.author}: [call {fc.name}] {fc.args}")
    return "\n".join(lines)


def build_compaction_prompt(events: list[Event]) -> str:
    previous_summary: str | None = None
    history_events = events
    if events and _is_prior_summary_seed(events[0]):
        seed_text = events[0].content.parts[0].text or ""
        previous_summary = _strip_banner(seed_text)
        history_events = events[1:]

    history = _format_history(history_events)
    sections: list[str] = [
        "Summarize the conversation history into the structured template below.",
        _TEMPLATE,
        _RULES,
    ]
    if previous_summary:
        sections.append(
            "\n".join(
                [
                    _MERGE_INSTRUCTIONS,
                    "<previous-summary>",
                    previous_summary,
                    "</previous-summary>",
                ]
            )
        )
    sections.append("Conversation history:\n" + history)
    return "\n\n".join(sections)


class HorizonSummarizer(LlmEventSummarizer):
    async def maybe_summarize_events(
        self, *, events: list[Event]
    ) -> Event | None:
        if not events:
            return None
        self._spawn_flush_fork_best_effort(events)

        prompt = build_compaction_prompt(events)
        llm_request = LlmRequest(
            model=self._llm.model,
            contents=[Content(role="user", parts=[Part(text=prompt)])],
        )
        summary_content: Content | None = None
        summary_usage_metadata = None
        async for llm_response in self._llm.generate_content_async(
            llm_request, stream=False
        ):
            if llm_response.content:
                summary_content = llm_response.content
                summary_usage_metadata = llm_response.usage_metadata
                break
        if summary_content is None:
            return None
        summary_content.role = "model"
        self._prepend_banner(summary_content)

        return Event(
            author="user",
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=events[0].timestamp,
                    end_timestamp=events[-1].timestamp,
                    compacted_content=summary_content,
                )
            ),
            invocation_id=Event.new_id(),
            usage_metadata=summary_usage_metadata,
        )

    @staticmethod
    def _spawn_flush_fork_best_effort(events: list[Event]) -> None:
        ctx = get_compaction_context()
        if ctx is None:
            return
        try:
            spawn_flush_fork(
                parent_memory_service=ctx.memory_service,
                parent_app_name=ctx.app_name,
                parent_user_id=ctx.user_id,
                events=events,
                sibling_agent_plugin=ctx.sibling_agent_plugin,
            )
        except Exception:
            logger.exception("summarizer: pre-compaction flush spawn failed")

    @staticmethod
    def _prepend_banner(content: Content) -> None:
        existing_text = ""
        if content.parts:
            first = content.parts[0]
            if first.text:
                existing_text = first.text
        banner_text = (
            f"{SUMMARY_BANNER_PREFIX}\n\n{existing_text}".rstrip()
            if existing_text
            else SUMMARY_BANNER_PREFIX
        )
        content.parts = [Part(text=banner_text), *(content.parts or [])[1:]]
