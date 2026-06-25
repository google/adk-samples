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

from unittest.mock import patch

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_ae_oauth.agent import root_agent


def test_agent_stream_no_auth() -> None:
    """
    Integration test: verifies the agent streams a response when asked
    to read a Drive file but no OAuth token is available.
    We mock negotiate_creds to return a pending dict so the agent
    responds with an authentication message rather than crashing.
    """

    session_service = InMemorySessionService()

    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Read the file with ID abc123")],
    )

    # Mock negotiate_creds to simulate "no OAuth token available"
    # so the test doesn't need real OAuth credentials
    with patch(
        "adk_ae_oauth.tools.negotiate_creds",
        return_value={"pending": True, "message": "Awaiting user authentication"},
    ):
        events = list(
            runner.run(
                new_message=message,
                user_id="test_user",
                session_id=session.id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )

    assert len(events) > 0, "Expected at least one event"

    has_text_content = False
    for event in events:
        if (
            event.content
            and event.content.parts
            and any(part.text for part in event.content.parts)
        ):
            has_text_content = True
            break
    assert has_text_content, "Expected at least one event with text content"
