import asyncio

# from functools import partial
from uuid import uuid4

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.run_config import RunConfig

# from google.adk.agents.workflow.join_node import JoinNode
# from google.adk.agents.workflow.function_node import FunctionNode
from google.adk.workflow import Workflow
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai.types import Content, ModelContent, Part

# Local sub-agent and node imports
from .agent import root_agent


class WorkflowDriver:
    def __init__(self):
        self.session_service = InMemorySessionService()

    async def run_workflow(
        self, agent: Workflow, user_input: str, session: Session
    ) -> None:
        """Helper to run a workflow and print its streaming output."""
        print(f"\n--- Running workflow: {agent.name} ---")

        user_content = Content(parts=[Part(text=user_input)])
        invocation_context = InvocationContext(
            session_service=self.session_service,
            agent=agent,
            invocation_id=str(uuid4()),
            session=session,
            user_content=user_content,
            run_config=RunConfig(),
        )

        async for event in agent.run_async(parent_context=invocation_context):
            if isinstance(event, ModelContent):
                if event.parts:
                    print(f"AGENT_OUTPUT: {event.parts[0].text}")


async def main() -> None:
    """Runs the two-part agentic workflow using a persistent session."""
    app_name = "Blog Agent"
    user_id = str(uuid4())

    driver = WorkflowDriver()
    session = await driver.session_service.create_session(
        app_name=app_name, user_id=user_id
    )

    # Run the first workflow to get the research report.
    await driver.run_workflow(
        agent=root_agent,
        user_input="The Future of AI in Education",
        session=session,
    )

    print(f"\nSESSION_STATE after research: {session.state}\n")

    # # Run the second workflow to write and publish the blog.
    # await driver.run_workflow(
    #     agent=blog_workflow,
    #     user_input="AI will personalize learning for every student.",
    #     session=session,
    # )


if __name__ == "__main__":
    print("Starting ADK 2.0 Workflow Demonstration...")
    try:
        asyncio.run(main())
    except ImportError as e:
        print(f"\nImportError: {e}")
        print("Please ensure the google_adk library is installed correctly.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
