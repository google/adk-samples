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


from google.adk.agents import LlmAgent
from google.adk.models import Gemini

from . import prompts
from .tools import dag_converter, knowledge_builder

# Agent for the knowledge base update workflow
knowledge_base_agent = LlmAgent(
    name="knowledge_base_updater",
    model=Gemini(model="gemini-2.5-pro"),
    description="A workflow to update the RAG knowledge base for Airflow migrations",
    instruction=prompts.KNOWLEDGE_BASE_AGENT_PROMPT,
    tools=[knowledge_builder.run_pipeline],
)

# The main agent that interacts with the user
root_agent = LlmAgent(
    name="airflow_migration_assistant",
    description="An assistant to help migrate Airflow DAGs between versions. Use tools and agents provided to accomplish this.",
    instruction=prompts.MIGRATION_ASSISTANT_PROMPT,
    model=Gemini(model="gemini-2.5-pro"),
    sub_agents=[knowledge_base_agent],
    tools=[dag_converter.convert_dags],
)
