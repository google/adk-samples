import asyncio

from blogger_agent.agent import root_agent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# Load env variables for direct execution
load_dotenv()


async def main():
    """Runs the agent with a sample query."""
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="app", user_id="test_user", session_id="test_session"
    )
    runner = Runner(agent=root_agent, app_name="app", session_service=session_service)

    queries = [
        "I want to write a blog post about the new features in the latest version of the ADK.",
        "looks good, write it",
        "1",
        "looks good, I approve",
        "yes",
        "my_new_blog_post.md",
    ]

    for query in queries:
        print(f">>> {query}")
        async for event in runner.run_async(
            user_id="test_user",
            session_id="test_session",
            new_message=genai_types.Content(
                role="user", parts=[genai_types.Part.from_text(text=query)]
            ),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())
