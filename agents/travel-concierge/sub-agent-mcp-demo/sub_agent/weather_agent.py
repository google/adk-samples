from . import prompt
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv


load_dotenv()

weather_tool = []

create_weather = Agent(
    model=LiteLlm(
        model="azure/gpt-4o",
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION")
    ),
    name="weather_agent",
    description="Provides weather forecasts using MCP tools.",
    instruction=prompt.WEATHER_INSTR_TEMPLATE,
    tools= weather_tool
)