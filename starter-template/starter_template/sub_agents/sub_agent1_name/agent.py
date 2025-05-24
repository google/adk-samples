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

from .tools.example_tool2 import tool2_name
from .prompt import SUB_AGENT1_PROMPT

# ----- Define a specialized sub-agent to be used by root agent -----

sub_agent1_name = Agent(
    name="sub_agent1",
    model="gemini-2.0-flash",
    instruction=SUB_AGENT1_PROMPT,
    tools=[
        # Custom tools are defined in tools.py
        # Tool 1
        tool2_name,
        # Tool 3
    ],
)
