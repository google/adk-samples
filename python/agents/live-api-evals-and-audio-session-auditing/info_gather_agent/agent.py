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
from utils.config import LIVE_AGENT_MODEL
from utils.prompt import BASE_SYSTEM_INSTRUCTION

root_agent = Agent(
    name="info_gather_agent",
    model=LIVE_AGENT_MODEL,
    instruction=BASE_SYSTEM_INSTRUCTION,
)
