import os

from google import genai
from google.adk.agents import Agent
from google.adk.tools import load_artifacts  # type: ignore

from .common.llm import GeminiWithLocation
from .common.utils import load_prompt
from .config import config
from .visualization_tools import (
    execute_visualization_code,
)

visualization_agent = Agent(
    model=GeminiWithLocation(
        model=config.agent_settings.model, location=config.GOOGLE_GENAI_LOCATION
    ),
    name="visualization_agent",
    description="Agent specialized in creating interactive and static visualizations.",
    instruction=load_prompt(
        os.path.dirname(__file__), "visualization_agent.txt"
    ),
    tools=[execute_visualization_code, load_artifacts],
    generate_content_config=genai.types.GenerateContentConfig(
        max_output_tokens=config.VISUALIZATION_AGENT_MAX_OUTPUT_TOKENS,
    ),
)
