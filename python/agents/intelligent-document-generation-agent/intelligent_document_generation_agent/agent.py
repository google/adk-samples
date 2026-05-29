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

import os

import vertexai
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.models.google_llm import Gemini
from google.genai.types import HttpRetryOptions

from .subagents import unified_processing_agent
from .utils.config import settings
from .utils.logging_setup import setup_logging

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"

setup_logging()

vertexai.init(
    project=settings.GOOGLE_CLOUD_PROJECT,
    location=settings.GOOGLE_CLOUD_LOCATION,
    staging_bucket=settings.ADK_STAGING_BUCKET,
)


def session_service_builder():
    from google.adk.sessions import VertexAiSessionService

    return VertexAiSessionService(
        project=settings.GOOGLE_CLOUD_PROJECT, location=settings.GOOGLE_CLOUD_LOCATION
    )


ROOT_AGENT_INSTRUCTIONS = """
You are a world-class AI assistant acting as an expert data analyst. Your goal is to process uploaded project documents and generate project summaries.

Follow these explicit steps:
1. **Acknowledge Upload**: The system will automatically save the user's uploaded PDF files as artifacts. You do not need to call any tool for this. Acknowledge that the files have been received and inform the user that you are beginning the analysis, which may take a few minutes.
2. **Process Documents**: Call the `unified_processing_agent`. This agent will automatically extract the necessary information and generate the required markdown documents. Once finished, inform the user that the analysis and generation are complete.
"""

retry_options = HttpRetryOptions(
    attempts=10, initial_delay=10, max_delay=5000, jitter=0.5
)

root_agent = LlmAgent(
    name="document_generation_agent_demo",
    model=Gemini(model=settings.WORKER_MODEL, retry_options=retry_options),
    description="A top-level agent that oversees document extraction and markdown generation.",
    instruction=ROOT_AGENT_INSTRUCTIONS,
    sub_agents=[unified_processing_agent],
)

app = App(root_agent=root_agent, name="intelligent_document_generation_agent")
