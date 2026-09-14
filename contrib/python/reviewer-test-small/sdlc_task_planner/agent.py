from google.adk.agents import LlmAgent

from .config import config
from .prompt import get_prompt


root_agent = LlmAgent(
    name="sdlc_task_planner_agent",
    model=config.default_llm,
    description="Agent that manages task creation and outputs a detailed MR plan based on technical design.",
    instruction=get_prompt(),
    tools=[],
)
