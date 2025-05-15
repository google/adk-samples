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

from google.adk.agents import Agent

from dotenv import load_dotenv

from .prompt import ROOT_PROMPT
from .tools.example_tool import tool1_name

from .sub_agents.sub_agent1_name import sub_agent1_name

load_dotenv()


# ----- Define Router (root) Agent -----


root_agent = Agent(
    name="starter_template",
    model="gemini-2.0-flash",
    instruction=ROOT_PROMPT,
    tools=[
        # Custom tools are defined in tools/ folder
        tool1_name,
        # Tool 2
        # Tool 3
    ],
    sub_agents=[
        # Sub agents are defined in sub_agents/ folder
        sub_agent1_name,
        # Agent 2
        # Agent 3
    ],
)
