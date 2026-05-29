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

import logging
from datetime import datetime
from typing import AsyncGenerator

import vertexai
from google.adk.agents import BaseAgent, InvocationContext, LlmAgent, ParallelAgent
from google.adk.events import Event
from google.adk.models.google_llm import Gemini
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from google.genai import types
from google.genai.types import HttpRetryOptions
from typing_extensions import override

from .callbacks import (
    after_extraction_callback,
    after_feature_list_generation_callback,
    after_security_overview_generation_callback,
    after_summary_generation_callback,
    before_extraction_callback,
)
from .resources.data_model import ExtractedInformation
from .utils.config import settings

vertexai.init(
    project=settings.GOOGLE_CLOUD_PROJECT,
    location="global",
    staging_bucket=settings.ADK_STAGING_BUCKET,
)

MODEL = Gemini(
    model=settings.WORKER_MODEL,
    retry_options=HttpRetryOptions(
        attempts=20, initial_delay=10, max_delay=10000, jitter=0.5
    ),
)

extraction_agent = LlmAgent(
    name="extraction_agent",
    model=MODEL,
    description="Extracts general project information from documents.",
    instruction="""You are an expert technical business analyst.
    Please review the provided source documents and extract the requested information.
    Your output MUST strictly conform to the ExtractedInformation schema.
    """,
    tools=[load_artifacts_tool],
    output_schema=ExtractedInformation,
    output_key="populated_data_model_json",
    before_agent_callback=before_extraction_callback,
    after_agent_callback=after_extraction_callback,
)

summary_generation_agent = LlmAgent(
    name="summary_generation_agent",
    model=MODEL,
    description="Generates a Markdown summary of the project.",
    instruction="""You are an expert technical writer.
    Using the extracted project information provided in the context, write a concise 1-page Markdown summary of the project, including its name, purpose, and target audience.

    Data:
    {populated_data_model_json}

    Output strictly as Markdown text.
    """,
    output_key="summary_md",
    after_agent_callback=after_summary_generation_callback,
)

feature_list_generation_agent = LlmAgent(
    name="feature_list_generation_agent",
    model=MODEL,
    description="Generates a Markdown document listing key features and technologies.",
    instruction="""You are an expert technical writer.
    Using the extracted project information, write a Markdown document that lists the key features, technologies used, and any external integrations in a structured way (bullet points, tables).

    Data:
    {populated_data_model_json}

    Output strictly as Markdown text.
    """,
    output_key="feature_list_md",
    after_agent_callback=after_feature_list_generation_callback,
)

security_overview_generation_agent = LlmAgent(
    name="security_overview_generation_agent",
    model=MODEL,
    description="Generates a Markdown document detailing data handled and security measures.",
    instruction="""You are an expert technical writer.
    Using the extracted project information, write a Markdown document summarizing the data handled by the system and outlining any security or privacy measures.

    Data:
    {populated_data_model_json}

    Output strictly as Markdown text.
    """,
    output_key="security_overview_md",
    after_agent_callback=after_security_overview_generation_callback,
)

parallel_document_generation_agent = ParallelAgent(
    name="parallel_document_generation_agent",
    description="Generates multiple output documents in parallel based on extracted information.",
    sub_agents=[
        summary_generation_agent,
        feature_list_generation_agent,
        security_overview_generation_agent,
    ],
)


class UnifiedProcessingAgent(BaseAgent):
    """
    A custom agent that extracts project information from documents,
    and then generates multiple output documents in parallel.
    """

    extraction_agent: LlmAgent
    parallel_document_generation_agent: ParallelAgent

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        extraction_agent: LlmAgent,
        parallel_document_generation_agent: ParallelAgent,
    ):
        super().__init__(
            name=name,
            extraction_agent=extraction_agent,
            parallel_document_generation_agent=parallel_document_generation_agent,
            sub_agents=[extraction_agent, parallel_document_generation_agent],
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        def _yield_message(message: str):
            """Helper to yield a message to the UI."""
            return Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=message)]),
            )

        # Step 1: Extraction
        yield _yield_message("\n🤖 Starting data extraction from documents...\n\n")
        logging.info(
            f"Starting data extraction at timestamp '{datetime.now().strftime('%Y%m%d-%H%M%S')}'"
        )
        async for event in self.extraction_agent.run_async(ctx):
            yield event

        # Step 2: Parallel Document Generation
        yield _yield_message("\n🤖 Generating output documents in parallel...\n\n")
        logging.info("Starting parallel document generation...")
        async for event in self.parallel_document_generation_agent.run_async(ctx):
            yield event

        # Step 3: Print URLs to user
        yield _yield_message(
            "\n🤖 Document generation complete. Here are your signed PDF URLs:\n"
        )

        # Access the session state inside the InvocationContext
        # Note: Depending on the ADK version, state is inside ctx.session.state
        session_state = ctx.session.state
        for doc_key, title in [
            ("summary_md", "Summary"),
            ("feature_list_md", "Feature List"),
            ("security_overview_md", "Security Overview"),
        ]:
            pdf_url = session_state.get(f"{doc_key}_pdf_url")
            if pdf_url:
                yield _yield_message(f"- **{title}**: [Download PDF]({pdf_url})\n")


unified_processing_agent = UnifiedProcessingAgent(
    name="unified_processing_agent",
    extraction_agent=extraction_agent,
    parallel_document_generation_agent=parallel_document_generation_agent,
)
