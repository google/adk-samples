# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Travel Conceirge Agent Grounded by Google Maps Platform MCP"""

import os
import pathlib
import re
import logging
from datetime import date
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset
from google.adk.tools.mcp_tool import McpToolset
from google.adk.planners import BuiltInPlanner
from google.genai import types
from typing import Optional, Dict, Any
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not GOOGLE_MAPS_API_KEY:
    raise ValueError("Missing GOOGLE_MAPS_API_KEY environment variable.")

current_date = date.today().strftime("%A, %B %d, %Y")

# ==========================================
# AGENT & TOOL SETUP
# ==========================================

BASE_SYSTEM_INSTRUCTION = f"""
You are a Premium Travel Orchestrator. Your sole purpose is to assist users with travel planning, location discovery, route mapping, weather checks, and culinary/cultural recommendations. The current system date is {current_date}.

# 1. OUT-OF-DOMAIN PROTOCOL (Strict Refusal)
You are strictly forbidden from answering queries unrelated to travel, geography, food, hospitality, or local experiences.
If a user asks about coding (e.g., Python bugs), math, writing essays, or general non-travel trivia:
- Politely decline.
- Explicitly state that your expertise is limited to travel and local discovery.
- Pivot by asking if they need help planning a trip or finding a great local spot.

# 2. TANGENTIAL KNOWLEDGE PROTOCOL (The "Tiramisu" Rule)
If a user asks a factual question about food, a cultural item, or a historical concept that *can* be tied to a physical location (e.g., "What is tiramisu?", "What is Gothic architecture?"):
- Provide a brief, helpful 1-2 sentence explanation of the concept.
- IMMEDIATELY pivot to your primary domain. Ask the user for their current location or target city so you can search for the best places to experience or eat that item.

# 3. TOOL EXECUTION BOUNDARIES
- NEVER call `lookup_weather` for general history, trivia, or factual questions.
- ONLY call `lookup_weather` if the user explicitly asks for the forecast, OR if they have confirmed they are actively planning an itinerary/trip for a specific date and destination.
"""

travel_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "travel-concierge"
)

maps_mcp_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://mapstools.googleapis.com/mcp",
        headers={
            "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
    )
)

my_skill_toolset = skill_toolset.SkillToolset(
    skills=[travel_skill],
    additional_tools=[maps_mcp_toolset]
)



root_agent = Agent(
    model='gemini-3-flash-preview',
    name='travel_planner_agent',
    description="A highly capable assistant leveraging specialized modular skills and spatial tools.",
    instruction=BASE_SYSTEM_INSTRUCTION,
    planner=BuiltInPlanner(        
        thinking_config=types.ThinkingConfig(
            include_thoughts=True
        )
    ),
    tools=[my_skill_toolset]
)