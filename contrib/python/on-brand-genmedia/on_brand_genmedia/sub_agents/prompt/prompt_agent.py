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

from ... import config
from ..tools.fetch_policy_tool import get_policy
from .prompt import PROMPT
from .tools.fetch_existing_assets import search_asset_bank

image_gen_prompt_generation_agent = Agent(
    name="image_gen_prompt_generation_agent",
    model=config.GENAI_MODEL,
    description=(
        "You are an expert in creating image generation prompts for a particular brand"
    ),
    instruction=(PROMPT),
    tools=[search_asset_bank, get_policy],
    output_key="image_gen_prompt",  # gets stored in session.state
)
