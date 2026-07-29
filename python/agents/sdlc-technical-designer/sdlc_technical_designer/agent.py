import logging

from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .config import config
from .prompt import get_prompt
from .tools.spanner_query_tools import SpannerQueryTools

logger = logging.getLogger(__name__)

tools_enabled = bool(
    config.spanner_project_id
    and config.spanner_instance_id
    and config.spanner_database_id
)
if tools_enabled:
    logger.info("Starting Technical Designer Agent with Spanner tools enabled.")
else:
    logger.info("Starting Technical Designer Agent without Spanner tools.")

instruction_text = get_prompt(tools_enabled=tools_enabled)


root_agent = LlmAgent(
    name="sdlc_technical_designer_agent",
    model=config.default_llm,
    description="Analyzes user stories and generates concrete, structured technical designs with recommendations.",
    instruction=instruction_text,
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=-1,
        )
    ),
    tools=[*SpannerQueryTools.get_toolset()] if tools_enabled else [],
)
