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

"""ADK agent definition for the retail product search skill.

Exposes a :data:`root_agent` (and :data:`app`) that uses Vertex AI Vector
Search 2.0 for semantic product retrieval via the :func:`retrieve_docs` tool.
"""

# pylint: disable=line-too-long
# (error messages and example URLs are intentionally long; breaking them hurts copy-paste UX.)

import os
import re

import google
import vertexai
from google.adk import agents, apps, models
from scripts.retrievers import search_collection

LLM_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
VECTOR_SEARCH_LOCATION = os.getenv("VECTOR_SEARCH_LOCATION", "us-central1")
LLM = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

_, _default_project = google.auth.default()
_project_id = os.getenv("GOOGLE_CLOUD_PROJECT", _default_project)
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", _project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LLM_LOCATION)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
vertexai.init(project=_project_id, location=LLM_LOCATION)


_COLLECTION_PATH_RE = re.compile(
    r"^projects/[^/\s]+/locations/[^/\s]+/collections/[^/\s]+$"
)


def _get_vector_search_collection() -> str:
    """Return the Vector Search collection resource path.

    Reads from the ``VECTOR_SEARCH_COLLECTION`` env var when set; otherwise
    builds the default path from ``GOOGLE_CLOUD_PROJECT`` and
    ``VECTOR_SEARCH_LOCATION``.

    Returns:
        The fully qualified Vector Search collection path.

    Raises:
        ValueError: If ``VECTOR_SEARCH_COLLECTION`` is set to a malformed
            path (e.g. contains a newline from a wrapped shell paste, which
            silently causes the Vector Search SDK to return a 501).
    """
    raw = os.getenv("VECTOR_SEARCH_COLLECTION")
    if raw is None:
        return (
            f"projects/{_project_id}/locations/{VECTOR_SEARCH_LOCATION}"
            "/collections/retail-skill-products-collection"
        )
    # Strip whitespace -- multi-line shell pastes can embed a newline mid-path,
    # which the Vector Search SDK silently maps to a 501 from the wrong endpoint.
    cleaned = raw.strip()
    if not _COLLECTION_PATH_RE.match(cleaned):
        raise ValueError(
            "VECTOR_SEARCH_COLLECTION is malformed -- expected "
            "'projects/<project>/locations/<region>/collections/<id>' with no whitespace. "
            f"Got: {raw!r}. "
            "Tip: type the export on a single line, e.g. "
            "export VECTOR_SEARCH_COLLECTION="
            '"projects/$GOOGLE_CLOUD_PROJECT/locations/us-central1/'
            'collections/retail-skill-products-collection"'
        )
    return cleaned


def retrieve_docs(query: str) -> str:
    """Search the product catalog using semantic similarity.

    Call this for every user query about products -- by name, by attributes
    (price/brand/category), by intent ('something for my desk'), or by
    comparison ('X vs Y'). Returns a formatted list of matching products
    with name, price, brand, rating, and a description snippet.

    Args:
        query: The product search query in natural language.

    Returns:
        Formatted product list, or a "no matches" message.
    """
    try:
        return search_collection(
            query=query,
            collection_path=_get_vector_search_collection(),
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return (
            f"Calling retrieval tool with query:\n\n{query}\n\n"
            f"raised the following error:\n\n{type(e)}: {e}"
        )


INSTRUCTION = """You are a retail product search assistant.
Use the retrieve_docs tool to search the product catalog for every user query.

IMPORTANT: After receiving tool results, you MUST immediately present the results to the user.
Do not stop after saying "let me search" -- always continue to present the full results.

When presenting search results:
- Say each product's name, price, brand, and a short description.
- Keep it conversational and concise.
- Never make up products -- only mention products returned by the tool.
- If no products match, say so and suggest broadening the search."""


root_agent = agents.Agent(
    name="root_agent",
    model=models.Gemini(model=LLM),
    instruction=INSTRUCTION,
    tools=[retrieve_docs],
)

app = apps.App(
    root_agent=root_agent,
    # ADK's `adk web` auto-names the app from the agent module's parent
    # directory. The agent lives at scripts/agent.py, so ADK names it
    # "scripts". The App name MUST match the auto-discovered name or
    # session creation fails with "Session not found: <id>. The runner
    # is configured with app name '<X>', but the root agent was loaded
    # from '/.../scripts'."
    name="scripts",
)
