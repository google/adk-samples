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

from app.retrievers import create_search_tool

# Load configuration from .env (see .env.example). Values already present in
# the environment win; we only fill in defaults and resolve the project from
# Application Default Credentials when they are not set.
load_dotenv()

LLM_LOCATION = "global"
LOCATION = "us-east1"
LLM = "gemini-flash-latest"

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LLM_LOCATION)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
if not project_id:
    _, project_id = google.auth.default()
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

vertexai.init(project=project_id, location=LOCATION)


data_store_region = os.getenv("DATA_STORE_REGION", "global")
# Collection that owns the data store. A GCS data connector creates the data
# store inside its own collection ("<project_name>-collection"), so set this
# from the `data_store_collection` Terraform output after `make setup-infra`.
data_store_collection = os.getenv("DATA_STORE_COLLECTION", "default_collection")
data_store_id = os.getenv(
    "DATA_STORE_ID", "rag-agent-search-collection_documents"
)
data_store_path = (
    f"projects/{project_id}/locations/{data_store_region}"
    f"/collections/{data_store_collection}/dataStores/{data_store_id}"
)

vertex_search_tool = create_search_tool(data_store_path)


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
    tools=[vertex_search_tool],
)

app = App(
    root_agent=root_agent,
    name="app",
)
