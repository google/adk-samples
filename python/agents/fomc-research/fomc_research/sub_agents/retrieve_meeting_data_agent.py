# Copyright 2025 Google LLC
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

"""Retrieve meeting data sub-agent for FOMC Research Agent"""

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from ..agent import MODEL
from ..shared_libraries.callbacks import rate_limit_callback
from ..tools.fetch_page import fetch_page_tool
from . import retrieve_meeting_data_agent_prompt
from .extract_page_data_agent import ExtractPageDataAgent

RetrieveMeetingDataAgent = Agent(
    model=MODEL,
    name="retrieve_meeting_data_agent",
    description=("Retrieve data about a Fed meeting from the Fed website"),
    instruction=retrieve_meeting_data_agent_prompt.PROMPT,
    tools=[
        fetch_page_tool,
        AgentTool(ExtractPageDataAgent),
    ],
    sub_agents=[],
    before_model_callback=rate_limit_callback,
)
