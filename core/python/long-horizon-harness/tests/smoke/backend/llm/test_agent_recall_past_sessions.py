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

"""LLM-driven: agent reaches for ``recall_past_sessions`` when asked about
a prior conversation that's not in preloaded memory.

Seeds a sibling session for the same user via the shared
``InMemorySessionService``, then asks the active session a question that
only makes sense if the agent looks at the other session's transcript.
"""

from __future__ import annotations

import pytest
from google.adk.events.event import Event
from google.genai import types as genai_types

from tests.smoke.backend.llm.conftest import function_call_names

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _seed_sibling_session(
    *, runner, user_id: str, user_text: str, agent_text: str
) -> str:
    svc = runner.session_service
    session = await svc.create_session(
        app_name=runner.app_name, user_id=user_id
    )
    await svc.append_event(
        session,
        Event(
            invocation_id="seed",
            author="user",
            content=genai_types.Content(
                role="user", parts=[genai_types.Part(text=user_text)]
            ),
        ),
    )
    await svc.append_event(
        session,
        Event(
            invocation_id="seed",
            author="root_agent",
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=agent_text)]
            ),
        ),
    )
    return session.id


async def test_agent_invokes_recall_past_sessions_on_prior_chat_question(
    llm_runner,
) -> None:
    await _seed_sibling_session(
        runner=llm_runner.runner,
        user_id=llm_runner.user_id,
        user_text="please build me a small tetris game in python",
        agent_text="Done — wrote tetris.py with a 20x10 grid and arrow-key controls.",
    )

    events = await llm_runner.run_once(
        "In a previous conversation I asked you to build something for me. "
        "Look at my other sessions and tell me what it was."
    )

    names = function_call_names(events)
    assert "recall_past_sessions" in names, (
        f"expected agent to call recall_past_sessions; got: {names}"
    )
