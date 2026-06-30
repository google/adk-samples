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

import pytest
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def _has_gcp_credentials() -> bool:
    """Return True if Application Default Credentials are available."""
    try:
        import google.auth

        google.auth.default()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _has_gcp_credentials(),
    reason="GCP credentials (ADC) required for the live Gemini call",
)
def test_agent_stream() -> None:
    """
    Integration test for the agent stream functionality.
    Tests that the agent returns valid streaming responses.

    The retriever (search_collection) is mocked via INTEGRATION_TEST=TRUE, but
    the agent still makes a live Gemini call, so this requires GCP credentials.
    """
    # Imported here so credential-less environments skip rather than error
    # at import time (app.agent calls google.auth.default() on import).
    from app.agent import root_agent

    session_service = InMemorySessionService()

    session = session_service.create_session_sync(
        user_id="test_user", app_name="test"
    )
    runner = Runner(
        agent=root_agent, session_service=session_service, app_name="test"
    )

    message = types.Content(
        role="user", parts=[types.Part.from_text(text="Why is the sky blue?")]
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one message"

    has_text_content = False
    for event in events:
        if (
            event.content
            and event.content.parts
            and any(part.text for part in event.content.parts)
        ):
            has_text_content = True
            break
    assert has_text_content, "Expected at least one message with text content"
