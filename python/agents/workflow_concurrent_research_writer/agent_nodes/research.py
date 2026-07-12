from google.adk.agents.llm_agent import LlmAgent
from ..prompts import (
    JOIN_AND_DISTILL_PROMPT,
)
import asyncio
from ..tools import execute_search
from google.adk.workflow import FunctionNode

import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part


# The LlmAgent that will be run in parallel for each platform.
_research_worker_llm_agent = LlmAgent(
    # The name for the inner agent is not strictly necessary for the
    # workflow but can be useful for debugging.
    name="research_worker_llm_agent",
    model="gemini-2.5-flash",
    instruction="""Your sole task is to research the topic '{topic}' on a specific platform.
The platform you MUST use is provided as your input. Your entire input is the name of the platform.
DO NOT ask for the platform. Use the input you are given.
Execute a search on that platform for the topic and summarize the results.""",
    tools=[execute_search],
)


from pydantic import BaseModel

class DistillInput(BaseModel):
    research: str

# The Synthesizer joins the results from the parallel workers into a final report.
distill_agent = LlmAgent(
    name="join_and_distill_agent",
    model="gemini-2.5-flash",
    input_schema=DistillInput,
    instruction=JOIN_AND_DISTILL_PROMPT,
    output_schema=str
)



research_worker_agent = LlmAgent(
    name="research_worker_agent",
    model="gemini-2.5-flash",
    instruction="""
Research topic '{topic}' on the given platform.

The platform is provided as input.

Use execute_search to search that platform.

Summarize the results.
""",
    tools=[execute_search],
)

APP_NAME = "parallel_research"

session_service = InMemorySessionService()

research_runner = Runner(
    app_name=APP_NAME,
    agent=research_worker_agent,
    session_service=session_service,
)

async def run_research_agent(topic, platform) -> str:
    """
    Executes research_worker_agent for a single platform.

    Returns the final generated text.
    """

    user_id = "parallel-worker"

    session_id = str(uuid.uuid4())

    # Every parallel invocation gets its own session
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={
            "topic": topic,
        },
    )

    user_message = Content(
        role="user",
        parts=[
            Part.from_text(text=platform),
        ],
    )

    print(f"user_message -- {user_message}")

    final_text = ""

    async for event in research_runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        if event.is_final_response():
            if (
                event.content
                and event.content.parts
                and event.content.parts[0].text
            ):
                final_text = event.content.parts[0].text
    print(f"final text -- \n\n {final_text}")
    return final_text

from google.genai.types import Content, Part


async def parallel_research(
    node_input: list[str],
    topic:str
) -> DistillInput:
    # print(f"Kushagra -- topic : {topic} ")
    platforms_to_research = node_input
    print(f"Kushagra -- topic : {topic} ")
    reports = await asyncio.gather(
        *[
            run_research_agent(topic, platform)
            for platform in platforms_to_research
        ]
    )

    combined = "\n\n---\n\n".join(reports)
    print(f"Combined -- {combined}")
    return DistillInput(
        research=combined
    )
