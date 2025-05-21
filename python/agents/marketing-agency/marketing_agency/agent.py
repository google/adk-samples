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

"""Marketing_coordinator Agent assists in creating effective online content."""

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from . import prompt
from .sub_agents.domain_create import domain_create_agent
from .sub_agents.logo_create import logo_create_agent
from .sub_agents.marketing_create import marketing_create_agent
from .sub_agents.website_create import website_create_agent

MODEL = "gemini-2.5-pro-preview-05-06" 

marketing_coordinator = LlmAgent(
    name="marketing_coordinator",
    model=MODEL,
    description=(
        "Establish a powerful online presence and connect with your audience "
        "effectively. Guide you through defining your digital identity, from "
        "choosing the perfect domain name and crafting a professional "
        "website, to strategizing online marketing campaigns, "
        "designing a memorable logo, and creating engaging short videos"
    ),
    instruction=prompt.MARKETING_COORDINATOR_PROMPT,
    tools=[
        AgentTool(agent=domain_create_agent),
        AgentTool(agent=website_create_agent),
        AgentTool(agent=marketing_create_agent),
        AgentTool(agent=logo_create_agent),
    ],
)

root_agent = marketing_coordinator
