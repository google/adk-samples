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

"""marketing_create_agent: for creating marketing strategies"""

from google.adk import Agent

from . import prompt

MODEL = "gemini-2.5-pro-preview-05-06" 

marketing_create_agent = Agent(
    model=MODEL,
    name="marketing_create_agent",
    instruction=prompt.MARKETING_CREATE_PROMPT,
    output_key="marketing_create_output",
)
