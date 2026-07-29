from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from youtube_analyst import agent


def test_agent_stream() -> None:
    """
    Integration test for the agent stream functionality.
    Tests that the agent returns valid streaming responses.
    """

    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()

    session = session_service.create_session_sync(
        user_id="test_user", app_name="test"
    )
    runner = Runner(
        agent=agent.root_agent, 
        session_service=session_service, 
        artifact_service=artifact_service,
        app_name="test"
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="What are the latest videos from Google Cloud channel?")],
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
