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

"""Defines Search Results Agent using ADK ComputerUseToolset.

Follows official ADK Computer Use pattern:
https://github.com/google/adk-python/tree/main/contributing/samples/multimodal/computer_use
"""

import os

from google.adk.agents.llm_agent import Agent
from google.adk.tools.computer_use.computer_use_toolset import (
    ComputerUseToolset,
)

from .playwright_computer import PlaywrightComputer
from .prompt import SEARCH_RESULT_AGENT_PROMPT

computer = PlaywrightComputer()

search_results_agent = Agent(
    model=os.getenv("MODEL_NAME"),
    name="search_results_agent",
    description=(
        "Inspects search engine result pages (SERPs) and audits brand "
        "ranking visibility using ADK Computer Use Toolset."
    ),
    instruction=SEARCH_RESULT_AGENT_PROMPT,
    tools=[ComputerUseToolset(computer=computer)],
)
