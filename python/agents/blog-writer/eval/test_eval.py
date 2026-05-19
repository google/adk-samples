import pytest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from blogger_agent.agent import root_agent
from google.genai import types as genai_types


@pytest.mark.asyncio
async def test_blogger_agent_smoke_test():
    """
    Basic smoke test to verify the blogger agent can start and respond.
    """
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="app", user_id="test_user", session_id="test_session"
    )
    runner = Runner(
        agent=root_agent, app_name="app", session_service=session_service
    )

    # Test initial greeting/query
    query = "Hello, I want to write a blog post about Python testing best practices."
    response_received = False

    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=genai_types.Content(
            role="user", parts=[genai_types.Part.from_text(text=query)]
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text
            # Verify we got a meaningful response
            assert response_text is not None
            assert len(response_text) > 0
            response_received = True
            break

    assert response_received, "Agent did not provide a response"
