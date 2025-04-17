from . import prompt
import os

from google.adk.agents import Agent
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

create_planning = Agent(
    model=LiteLlm(
        model="azure/gpt-4o",
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION")
    ),
    name="planning_agent",
    description="""
     Responsible for planning and creating transportation or travel reservations
     (such as flights, high-speed rail, or shuttle services) based on the user's 
     itinerary and preferences, ensuring the user arrives at the destination smoothly and on time.
     """,
    instruction=prompt.PLANNING_INSTR_TEMPLATE,
    output_key="place",
)

