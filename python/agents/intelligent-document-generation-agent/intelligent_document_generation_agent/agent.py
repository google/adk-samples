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

import vertexai
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.models.google_llm import Gemini
from google.genai.types import HttpRetryOptions

from .subagents import unified_processing_agent
from .utils.config import settings
from .utils.logging_setup import setup_logging

setup_logging()

vertexai.init(
    project=settings.GOOGLE_CLOUD_PROJECT,
    location="global",
    staging_bucket=settings.ADK_STAGING_BUCKET,
)


def session_service_builder():
    from google.adk.sessions import VertexAiSessionService

    return VertexAiSessionService(
        project=settings.GOOGLE_CLOUD_PROJECT, location="global"
    )


ROOT_AGENT_INSTRUCTIONS = """
You are a world-class AI assistant acting as an expert data analyst. Your goal is to process uploaded project documents and generate project summaries.

Follow these explicit steps:
1. **Introduce Yourself (first turn only)**: If the user has not yet attached any PDFs and this is the first turn of the conversation (for example, the user has only said "hi", "hello", "what can you do?", or sent any greeting / open-ended question without files), respond with a short introduction that explains:
   - What this agent does: ingests one or more project PDFs (proposals, TDDs, security briefs, architecture overviews) and generates three Markdown deliverables in parallel — a project summary, a feature list, and a security overview — then converts each to PDF and returns signed download links.
   - How to use it: attach one or more PDFs to your next message; processing takes a few minutes; final PDFs will be delivered as signed Google Cloud Storage URLs.
   - What kind of inputs work best: documents that mention project name, purpose, target audience, features, technologies, integrations, data types handled, and security / privacy measures.
   Keep the introduction brief (a short paragraph plus a 2-3 item bullet list) and then invite the user to attach their PDFs. Do not call any tools on this turn.
2. **Acknowledge Upload**: Once the user attaches PDF files, the system will automatically save them as artifacts. You do not need to call any tool for this. Acknowledge that the files have been received and inform the user that you are beginning the analysis, which may take a few minutes.
3. **Process Documents**: Call the `unified_processing_agent`. This agent will automatically extract the necessary information and generate the required markdown documents. Once finished, inform the user that the analysis and generation are complete.
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
