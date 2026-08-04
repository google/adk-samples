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

from vertexai.preview.reasoning_engines import AdkApp
from economic_research.agent import ERAAgent

# Instantiate the agent
era_instance = ERAAgent()
root_agent = era_instance.get_app().root_agent

# Expose agent_runtime for agents-cli introspection
agent_runtime = AdkApp(
    agent=root_agent,
    enable_tracing=True,
)
