# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents import Agent

from . import config
from .prompt import CHECKER_PROMPT
from .tools.loop_condition_tool import check_tool_condition

# This agent is responsible for checking conditions and validating the scoring process
# It uses the check_tool_condition tool to evaluate whether the scoring process should continue
# The agent's output is stored in the "checker_output" key
checker_agent_instance = Agent(
    name="checker_agent",
    model=config.GENAI_MODEL,
    instruction=CHECKER_PROMPT,
    tools=[check_tool_condition],
    output_key="checker_output",
)
