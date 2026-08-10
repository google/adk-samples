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

"""Unit tests for the ``cancel()`` override on ``_TaskTrackingExecutor``.

ADK's ``A2aAgentExecutor.cancel`` raises ``NotImplementedError``, which aborts
a2a's ``on_cancel_task`` before it can cancel the running producer task. We
override ``cancel`` to emit a terminal ``canceled`` status so ``on_cancel_task``
proceeds. These tests pin that contract; no LLM is exercised.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.types import TaskState
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor

from horizon.a2a.executor import build_executor
from horizon.auth.identity import _user_id_var


@pytest.fixture(autouse=True)
def _authenticated_user():
    token = _user_id_var.set("u")
    try:
        yield
    finally:
        _user_id_var.reset(token)


def _make_context(*, task_id: str, session_id: str = "s") -> MagicMock:
    context = MagicMock()
    context.task_id = task_id
    context.context_id = session_id
    return context


@pytest.mark.asyncio
async def test_cancel_enqueues_terminal_canceled_status() -> None:
    executor = build_executor(runner=MagicMock())
    context = _make_context(task_id="task-cancel", session_id="s-cancel")
    event_queue = MagicMock()
    event_queue.enqueue_event = AsyncMock()

    await executor.cancel(context, event_queue)

    event_queue.enqueue_event.assert_awaited_once()
    event = event_queue.enqueue_event.await_args.args[0]
    # a2a 1.x has no per-event `final` flag; a canceled status IS terminal.
    assert event.status.state == TaskState.TASK_STATE_CANCELED
    assert event.task_id == "task-cancel"
    assert event.context_id == "s-cancel"


@pytest.mark.asyncio
async def test_cancel_does_not_raise() -> None:
    executor = build_executor(runner=MagicMock())
    context = _make_context(task_id="task-x")
    event_queue = MagicMock()
    event_queue.enqueue_event = AsyncMock()

    # The whole point of the override: it must NOT raise (the parent does).
    await executor.cancel(context, event_queue)


@pytest.mark.asyncio
async def test_base_executor_cancel_raises_not_implemented() -> None:
    """Regression guard documenting why the override exists: the ADK base
    ``cancel`` raises ``NotImplementedError`` when no executor impl is set,
    which would abort a2a's ``on_cancel_task`` before the producer is cancelled.
    """
    executor = build_executor(runner=MagicMock())
    executor._executor_impl = None
    context = _make_context(task_id="task-y")
    event_queue = MagicMock()
    event_queue.enqueue_event = AsyncMock()

    with pytest.raises(NotImplementedError):
        await A2aAgentExecutor.cancel(executor, context, event_queue)
