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

"""Shared Memory Bank configuration for both deployment targets."""

from vertexai._genai.types import (
    ManagedTopicEnum,
)
from vertexai._genai.types import (
    MemoryBankCustomizationConfig as CustomizationConfig,
)
from vertexai._genai.types import (
    MemoryBankCustomizationConfigMemoryTopic as MemoryTopic,
)
from vertexai._genai.types import (
    MemoryBankCustomizationConfigMemoryTopicManagedMemoryTopic as ManagedMemoryTopic,
)
from vertexai._genai.types import (
    ReasoningEngineContextSpecMemoryBankConfig as MemoryBankConfig,
)

# --- Memory Bank ---
# Define which memory topics the Memory Bank should extract and persist.
# This config is used by both deployment targets:
#   - Agent Engine: passed via context_spec in AgentEngineConfig (deploy.py)
#   - Cloud Run: passed via context_spec when creating Agent Engine (fast_api_app.py)
#
# Available managed topics:
#   USER_PERSONAL_INFO     - names, relationships, hobbies, important dates
#   USER_PREFERENCES       - likes, dislikes, preferred styles
#   KEY_CONVERSATION_DETAILS - milestones, task outcomes
#   EXPLICIT_INSTRUCTIONS  - things the user asks the agent to remember/forget
#
# You can also define custom topics with CustomMemoryTopic(label=..., description=...).
# See: https://docs.cloud.google.com/agent-builder/agent-engine/memory-bank/set-up#customization-config
memory_bank_config = MemoryBankConfig(
    customization_configs=[
        CustomizationConfig(
            memory_topics=[
                MemoryTopic(
                    managed_memory_topic=ManagedMemoryTopic(
                        managed_topic_enum=ManagedTopicEnum.USER_PERSONAL_INFO,
                    ),
                ),
                MemoryTopic(
                    managed_memory_topic=ManagedMemoryTopic(
                        managed_topic_enum=ManagedTopicEnum.USER_PREFERENCES,
                    ),
                ),
                MemoryTopic(
                    managed_memory_topic=ManagedMemoryTopic(
                        managed_topic_enum=ManagedTopicEnum.EXPLICIT_INSTRUCTIONS,
                    ),
                ),
            ],
        ),
    ],
)
