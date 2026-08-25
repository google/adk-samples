# Copyright 2025 Google LLC
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

"""Database Agent: get data from database (BigQuery) using NL2SQL."""

import logging
import os
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
from google.genai import types

from ...utils.utils import USER_AGENT
from . import tools
from .chase_sql import chase_db_tools
from .prompts import return_instructions_bigquery

logger = logging.getLogger(__name__)

NL2SQL_METHOD = os.getenv("NL2SQL_METHOD", "BASELINE")

# BigQuery built-in tools in ADK
# https://google.github.io/adk-docs/tools/built-in-tools/#bigquery
ADK_BUILTIN_BQ_EXECUTE_SQL_TOOL = "execute_sql"


def setup_before_agent_call(callback_context: CallbackContext) -> None:
    """Setup the agent."""

    if "database_settings" not in callback_context.state:
        callback_context.state["database_settings"] = (
            tools.get_database_settings()
        )


def store_results_in_context(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict,
) -> dict | None:
    # We are setting a state for the data science agent to be able to use the
    # sql query results as context
    if tool.name == ADK_BUILTIN_BQ_EXECUTE_SQL_TOOL:
        if tool_response["status"] == "SUCCESS":
            tool_context.state["bigquery_query_result"] = tool_response["rows"]

    return None


bigquery_tool_filter = [ADK_BUILTIN_BQ_EXECUTE_SQL_TOOL]
bigquery_tool_config = BigQueryToolConfig(
    write_mode=WriteMode.BLOCKED, application_name=USER_AGENT
)
bigquery_toolset = BigQueryToolset(
    tool_filter=bigquery_tool_filter, bigquery_tool_config=bigquery_tool_config
)

bigquery_agent = LlmAgent(
    model=os.getenv("BIGQUERY_AGENT_MODEL", ""),
    name="bigquery_agent",
    instruction=return_instructions_bigquery(),
    tools=[
        (
            chase_db_tools.initial_bq_nl2sql
            if NL2SQL_METHOD == "CHASE"
            else tools.bigquery_nl2sql
        ),
        bigquery_toolset,
    ],
    before_agent_callback=setup_before_agent_call,
    after_tool_callback=store_results_in_context,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
