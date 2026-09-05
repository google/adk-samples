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
# limitations under the License.§

"""Agent module for the jailbreak classification."""

import logging
import warnings

from google.adk.agents.llm_agent import Agent
from ...config import Config
from . import prompt

warnings.filterwarnings("ignore", category=UserWarning, module=".*pydantic.*")

configs = Config()

logger = logging.getLogger(__name__)


jailbreak_classification_agent = Agent(
    model=configs.agent_settings.model,
    name="jailbreak_classification_agent",
    instruction=prompt.JAILBREAK_FILTER_INSTRUCTION,
    output_key="jailbreak_classification"
)
