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
import sys

import google.auth

# --- Configuration ---
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
)
from google.adk.tools.bigquery import BigQueryToolset

load_dotenv()

# Default to Vertex AI backend (necessary for environments like Cloud Run
# where .env files are not deployed), but allow developers to override
# this to "False" (AI Studio) in their local .env file.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

credentials, PROJECT_ID = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Fallback to env var if default auth doesn't have it (e.g. some service accounts)
if not PROJECT_ID:
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")

if not PROJECT_ID:
    print(
        "Warning: Could not determine GOOGLE_CLOUD_PROJECT from credentials or environment. Agent may fail.",
        file=sys.stderr,
    )

DATASET_ID = os.environ.get("BQ_ANALYTICS_DATASET_ID", "adk_agent_analytics")
TABLE_ID = os.environ.get("BQ_ANALYTICS_TABLE_ID", "agent_events")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1")

# --- Initialize the Plugin ---
plugins = []
bq_logging_plugin = BigQueryAgentAnalyticsPlugin(
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    table_id=TABLE_ID,
    location=LOCATION,
)
plugins.append(bq_logging_plugin)


# --- Initialize Tools & Model ---
tools = []
bigquery_toolset = BigQueryToolset()
tools.append(bigquery_toolset)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="agent_observability_bq",
    description="An agent that demonstrates observability via logging to Google BigQuery.",
    instruction="""
    You are a data analyst agent specializing in Google BigQuery.
    
    You have access to a BigQuery toolset that you can use to interact with BigQuery.
    
    When asked about data, you should:
    1. Understand the user's intent.
    2. Write a valid BigQuery Standard SQL query.
    3. Use the BigQuery toolset to execute the query.
    4. Provide a helpful, clear answer based on the results.
    
    Always format your SQL queries clearly before executing them, and explain the results plainly to the user.
    Remember to use fully qualified table names in BigQuery (project.dataset.table) if necessary.
    """,
    tools=tools,
)

# --- Create the App ---
app = App(
    name="agent_observability_bq",
    root_agent=root_agent,
    plugins=plugins,
)
