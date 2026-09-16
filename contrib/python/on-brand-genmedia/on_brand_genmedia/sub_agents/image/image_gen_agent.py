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
from .prompt import IMAGE_GEN_PROMPT
from .tools.image_generation_tool import generate_images

image_generation_agent = Agent(
    name="image_generation_agent",
    model=config.GENAI_MODEL,
    description=(
        f"You are an expert in creating images with {config.IMAGE_GEN_MODEL}"
    ),
    instruction=(IMAGE_GEN_PROMPT),
    tools=[generate_images],
    output_key="output_image",
)
