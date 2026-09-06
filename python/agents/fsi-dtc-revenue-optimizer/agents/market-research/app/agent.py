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

import asyncio
import datetime
import logging
import os
import pathlib

import google.auth
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.models import Gemini
from google.genai import types

logger = logging.getLogger(__name__)

load_dotenv(override=True)

# Dynamic year tokens used in prompt templates so trend references stay current.
_CURRENT_YEAR = datetime.date.today().year
_NEXT_YEAR = _CURRENT_YEAR + 1


def load_prompt(prompt_name: str) -> str:
    prompt_path = pathlib.Path(__file__).parent / "prompts" / prompt_name
    try:
        text = prompt_path.read_text()
        # Replace year placeholders so prompts always reference the current year.
        return text.replace("{current_year}", str(_CURRENT_YEAR)).replace(
            "{next_year}", str(_NEXT_YEAR)
        )
    except FileNotFoundError:
        logger.warning("Prompt file %s not found.", prompt_name)
        return ""


system_prompt = load_prompt("system_prompt.md")
system_instructions = load_prompt("system_instructions.md")
model_id = os.getenv("ADK_MODEL", "gemini-3-flash-preview")



# Use setdefault so .env values are respected; only fill in if not already set.
_, _default_project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", _default_project_id or "")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


async def deep_research(query: str) -> str:
    """Performs deep research on a topic using the Gemini Deep Research Agent.

    This is a long-running operation that autonomously plans, searches the web,
    reads sources, and synthesizes a detailed, cited research report.
    Use this tool when the user asks for in-depth analysis, market research,
    competitive landscape reviews, or any task requiring thorough investigation
    across multiple sources. Expect it to take a few minutes to complete.

    Args:
        query: The research question or topic to investigate in detail.

    Returns:
        A detailed, cited research report, or an error message.
    """

    # Use the Gemini API with an AI Studio API key (not a GCP Console key).
    # Docs: https://ai.google.dev/gemini-api/docs/interactions
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable is not set."

    client = genai.Client(
        api_key=api_key,
        vertexai=False,
        http_options=types.HttpOptions(api_version="v1beta"),
    )

    try:
        print(f"\n[DEBUG] Deep Research API Call Starting for query: {query}", flush=True)
        interaction = await client.aio.interactions.create(
            input=query,
            agent="deep-research-pro-preview-12-2025",
            background=True,
        )
        print(f"\n[DEBUG] Deep Research started successfully! Interaction ID: {interaction.id}", flush=True)
        logger.info("Deep Research started: %s", interaction.id)

        # Poll until completed or failed (up to ~5 minutes)
        for _ in range(30):
            await asyncio.sleep(10)
            interaction = await client.aio.interactions.get(interaction.id)
            if interaction.status == "completed":
                print(f"\n[DEBUG] Deep Research Completed! Returning text.", flush=True)
                return interaction.outputs[-1].text
            elif interaction.status == "failed":
                print(f"\n[DEBUG] Deep Research failed: {interaction.error}", flush=True)
                return f"Deep Research failed: {interaction.error}"

        return (
            "Deep Research is still running after 5 minutes. "
            f"Interaction ID: {interaction.id}. Please try again later."
        )
    except Exception as e:
        logger.exception("Deep Research error")
        return f"Error running Deep Research: {e}"


shared_model = Gemini(
    model=model_id,
    retry_options=types.HttpRetryOptions(attempts=3),
)

root_agent = LlmAgent(
    name="Market_Research_Agent",
    model=shared_model,
    description=(
        "A market research agent that performs deep web research using "
        "Gemini Deep Research to produce detailed, cited reports on "
        "millennial consumer trends, digital life insurance, and FS Corp market dynamics."
    ),
    instruction=f"{system_prompt}\n\n{system_instructions}",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=8192,
    ),
    tools=[
        deep_research,
    ],
)

# Enable resumability only in deployed environments (Cloud Run sets the PORT env var).
_is_deployed = bool(os.environ.get("K_SERVICE"))

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(enabled=_is_deployed),
)
