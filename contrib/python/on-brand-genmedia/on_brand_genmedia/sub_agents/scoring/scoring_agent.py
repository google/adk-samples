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
from .prompt import SCORING_PROMPT
from .tools.get_images_tool import get_image
from .tools.set_score_tool import set_score

scoring_agent = Agent(
    name="scoring_agent",
    model=config.GENAI_MODEL,
    description=(
        "You are an expert in evaluating and scoring images based on various criteria "
        "provided to you."
    ),
    instruction=(SCORING_PROMPT),
    output_key="scoring",
    tools=[get_policy, get_image, set_score],
)
