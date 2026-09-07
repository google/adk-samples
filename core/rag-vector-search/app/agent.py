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

import google.auth
import vertexai
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.retrievers import search_collection

# Load configuration from .env (see .env.example). Values already present in
# the environment win; we only fill in defaults and resolve the project from
# Application Default Credentials when they are not set.
load_dotenv()

LLM_LOCATION = "global"
LOCATION = "us-central1"
LLM = os.getenv("MODEL_NAME")

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LLM_LOCATION)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
if not project_id:
    _, project_id = google.auth.default()
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

vertexai.init(project=project_id, location=LOCATION)


vector_search_collection = os.getenv(
    "VECTOR_SEARCH_COLLECTION",
    f"projects/{project_id}/locations/{LOCATION}/collections/rag-vector-search-collection",
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
            collection_path=vector_search_collection,
        )
    except Exception as e:
        return (
            f"Calling retrieval tool with query:\n\n{query}\n\n"
            f"raised the following error:\n\n{type(e)}: {e}"
        )


instruction = (
    "You are an AI assistant for question-answering tasks.\n"
    "Answer to the best of your ability using the context provided.\n"
    "Leverage the Tools you are provided to answer questions.\n"
    "If you already know the answer to a question, you can respond "
    "directly without using the tools."
)


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model=LLM,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=instruction,
    tools=[retrieve_docs],
)

app = App(
    root_agent=root_agent,
    name="app",
)
