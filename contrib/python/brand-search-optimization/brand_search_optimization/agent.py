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

"""Defines Brand Search Optimization Agent."""

import os

from google.adk.agents.llm_agent import Agent

from . import prompt
from .sub_agents.comparison.agent import comparison_root_agent
from .sub_agents.keyword_finding.agent import keyword_finding_agent
from .sub_agents.search_results.agent import search_results_agent

root_agent = Agent(
    model=os.getenv("MODEL_NAME", "gemini-3.7-flash"),
    name="brand_search_optimization",
    description="A helpful assistant for brand search optimization using Computer Use.",
    instruction=prompt.ROOT_PROMPT,
    sub_agents=[
        keyword_finding_agent,
        search_results_agent,
        comparison_root_agent,
    ],
)
