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

"""Tests for ``HorizonSummarizer`` — banner wrapping + pre-compaction flush fork.

The summarizer's LLM is stubbed so these tests drive the real override
end-to-end without hitting Gemini. Real LLM behavior lives in evals.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.events.event import Event
from google.genai.types import Content, Part

from horizon.context.compaction_context import (
    CompactionContext,
    bind_compaction_context,
    clear_compaction_context,
)
from horizon.context.summarizer import (
    SUMMARY_BANNER_PREFIX,
    HorizonSummarizer,
)

pytestmark = pytest.mark.asyncio


def _user_event(text: str, *, ts: float = 1.0) -> Event:
    return Event(
        author="user",
        content=Content(role="user", parts=[Part(text=text)]),
        invocation_id=f"test-{ts}",
        timestamp=ts,
    )


def _build_summarizer(
    summary_text: str | None = "structured summary",
    *,
    usage_metadata: Any = None,
) -> HorizonSummarizer:
    class _StubLlm:
        model = "stub"

        async def generate_content_async(self, request, stream=False):
            from google.adk.models.llm_response import LlmResponse

            if summary_text is None:
                return
            yield LlmResponse(
                content=Content(role="model", parts=[Part(text=summary_text)]),
                usage_metadata=usage_metadata,
            )

    return HorizonSummarizer(llm=_StubLlm())  # type: ignore[arg-type]


@pytest.fixture()
def _scoped_compaction_context() -> Any:
    clear_compaction_context()
    yield
    clear_compaction_context()


# =========================================================================
# Banner wrapping
# =========================================================================


async def test_banner_prefixed_to_compacted_content(
    _scoped_compaction_context: None,
):
    summarizer = _build_summarizer("middle was about auth")

    result = await summarizer.maybe_summarize_events(
        events=[_user_event("hello")]
    )

    assert result is not None
    assert result.actions is not None
    assert result.actions.compaction is not None
    text = result.actions.compaction.compacted_content.parts[0].text
    assert text is not None
    assert text.startswith(SUMMARY_BANNER_PREFIX)
    assert "middle was about auth" in text


async def test_compaction_event_carries_usage_metadata(
    _scoped_compaction_context: None,
):
    from google.genai.types import GenerateContentResponseUsageMetadata

    usage = GenerateContentResponseUsageMetadata(total_token_count=321)
    summarizer = _build_summarizer("summary", usage_metadata=usage)

    result = await summarizer.maybe_summarize_events(
        events=[_user_event("hello")]
    )

    assert result is not None
    assert result.usage_metadata is usage
    assert result.usage_metadata.total_token_count == 321


async def test_no_banner_when_llm_returns_nothing(
    _scoped_compaction_context: None,
):
    summarizer = _build_summarizer(summary_text=None)

    result = await summarizer.maybe_summarize_events(events=[_user_event("hi")])

    assert result is None


async def test_empty_events_returns_none(
    _scoped_compaction_context: None,
):
    summarizer = _build_summarizer()

    result = await summarizer.maybe_summarize_events(events=[])

    assert result is None


# =========================================================================
# Pre-compaction flush fork
# =========================================================================


@pytest.fixture()
def captured_flush(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {"called": False, "kwargs": None}

    def _fake_spawn_flush_fork(**kwargs: Any) -> bool:
        captured["called"] = True
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(
        "horizon.context.summarizer.spawn_flush_fork",
        _fake_spawn_flush_fork,
    )
    return captured


async def test_flush_fork_spawned_when_context_present(
    captured_flush: dict[str, Any],
    _scoped_compaction_context: None,
):
    memory_service = SimpleNamespace()
    bind_compaction_context(
        CompactionContext(
            memory_service=memory_service,
            app_name="lha-test",
            user_id="user-42",
        )
    )
    summarizer = _build_summarizer()
    events = [_user_event("hello"), _user_event("world", ts=2.0)]

    await summarizer.maybe_summarize_events(events=events)

    assert captured_flush["called"] is True
    kwargs = captured_flush["kwargs"]
    assert kwargs["parent_memory_service"] is memory_service
    assert kwargs["parent_app_name"] == "lha-test"
    assert kwargs["parent_user_id"] == "user-42"
    assert kwargs["events"] == events


async def test_flush_fork_skipped_when_context_missing(
    captured_flush: dict[str, Any],
    _scoped_compaction_context: None,
):
    summarizer = _build_summarizer()

    result = await summarizer.maybe_summarize_events(events=[_user_event("hi")])

    assert captured_flush["called"] is False
    assert result is not None


async def test_flush_fork_failure_does_not_block_compaction(
    monkeypatch: pytest.MonkeyPatch,
    _scoped_compaction_context: None,
):
    def _boom(**_kwargs: Any) -> bool:
        raise RuntimeError("flush spawn exploded")

    monkeypatch.setattr("horizon.context.summarizer.spawn_flush_fork", _boom)
    bind_compaction_context(
        CompactionContext(
            memory_service=SimpleNamespace(),
            app_name="lha-test",
            user_id="user-42",
        )
    )
    summarizer = _build_summarizer()

    result = await summarizer.maybe_summarize_events(events=[_user_event("hi")])

    assert result is not None
    assert result.actions.compaction is not None
