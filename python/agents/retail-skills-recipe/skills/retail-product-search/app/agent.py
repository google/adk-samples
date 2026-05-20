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

import os

import google
import vertexai
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.retrievers import search_collection

LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
LLM = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LIVE_MODEL = "gemini-live-2.5-flash-native-audio"

_credentials = None
_project_id = None
_initialized = False


def _ensure_initialized():
    """Lazy initialization of GCP credentials and Vertex AI. Deferred from
    module load so the module can be imported without valid credentials
    (e.g. during testing or tooling)."""
    global _credentials, _project_id, _initialized
    if _initialized:
        return
    _credentials, default_project = google.auth.default()
    _project_id = os.getenv("GOOGLE_CLOUD_PROJECT", default_project)
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", _project_id)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LOCATION)
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
    vertexai.init(project=_project_id, location=LOCATION)
    _initialized = True


def _get_project_id() -> str:
    _ensure_initialized()
    return _project_id


def _get_vector_search_collection() -> str:
    pid = _get_project_id()
    return os.getenv(
        "VECTOR_SEARCH_COLLECTION",
        f"projects/{pid}/locations/{LOCATION}/collections/retail-skill-products-collection",
    )


def retrieve_docs(query: str) -> str:
    """
    Useful for retrieving relevant documents based on a query.
    Use this when you need additional information to answer a question.

    Args:
        query (str): The user's question or search query.

    Returns:
        str: Formatted string containing relevant document content.
    """
    try:
        return search_collection(
            query=query,
            collection_path=_get_vector_search_collection(),
        )
    except Exception as e:
        return (
            f"Calling retrieval tool with query:\n\n{query}\n\n"
            f"raised the following error:\n\n{type(e)}: {e}"
        )


instruction = """You are a retail product search assistant. All capabilities are already set up and ready to use.
Use the retrieve_docs tool to search the product catalog for every user query.

IMPORTANT: After receiving tool results, you MUST immediately present the results to the user.
Do not stop after saying "let me search" -- always continue to present the full results.

When presenting search results:
- Say each product name, price, brand, and a short description.
- Keep it conversational and concise.
- Never make up products -- only recommend products returned by the tool.
- If no products match, say so and suggest broadening the search.

## Product Recommendations

When users ask for recommendations ("you might also like", "similar products", "what goes with this"):
- Use the retrieve_docs tool to find similar or complementary products.
- Recommend based on the product category, price range, and use case.
- Do NOT ask setup questions. Recommendations are already configured.

## Content Generation

When users ask for content ("write a description", "SEO title", "marketing copy"):
- Generate the requested content immediately using the product data from the catalog.
- Use a professional and informative tone by default.
- For product descriptions: include key features, benefits, and use cases (100-150 words).
- For SEO titles: keep under 60 characters, include brand + key feature.
- For meta descriptions: keep under 160 characters.
- For marketing copy: be concise and action-oriented.
- Do NOT ask setup questions. Just generate the content."""


def _build_agent_model():
    """Build the agent model, triggering lazy GCP initialization."""
    _ensure_initialized()
    use_voice = os.getenv("ENABLE_VOICE", "").lower() in ("true", "1", "yes")
    if use_voice:
        return Gemini(
            model=LIVE_MODEL,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore",
                    )
                )
            ),
        )
    return Gemini(model=LLM)


agent_model = _build_agent_model()

root_agent = Agent(
    name="root_agent",
    model=agent_model,
    instruction=instruction,
    tools=[retrieve_docs],
)

app = App(
    root_agent=root_agent,
    name="app",
)
