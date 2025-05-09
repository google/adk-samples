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

"""Top level agent for data agent multi-agents.

-- it get data from database (e.g., BQ) using NL2SQL
-- then, it use NL2Py to do further data analysis as needed
"""
import os
from datetime import date

from google.genai import types

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import load_artifacts

from .sub_agents import bqml_agent
from .sub_agents.bigquery.tools import (
    get_database_settings as get_bq_database_settings,
)
from .prompts import return_instructions_root
from .tools import call_db_agent, call_ds_agent

date_today = date.today()


def setup_before_agent_call(callback_context: CallbackContext):
    """Setup the agent."""

    # setting up database settings in session.state
    if "database_settings" not in callback_context.state:
        db_settings = dict()
        db_settings["use_database"] = "BigQuery"  # Default or initial setting
        callback_context.state["all_db_settings"] = db_settings
    # Ensure all_db_settings is populated and consistent
    elif "all_db_settings" not in callback_context.state or \
         "use_database" not in callback_context.state["all_db_settings"]:
        # If database_settings exists but all_db_settings is missing or incomplete,
        # try to infer or default. This state might indicate an issue elsewhere.
        callback_context.state["all_db_settings"] = {"use_database": "BigQuery"} # Defaulting

    schema_info = "-- No schema information available for the configured database. --" # Default schema info

    if callback_context.state.get("all_db_settings", {}).get("use_database") == "BigQuery":
        # Get BQ settings and store them
        current_bq_settings = get_bq_database_settings()
        callback_context.state["database_settings"] = current_bq_settings

        # Safely get the DDL schema
        schema = current_bq_settings.get("bq_ddl_schema")
        if schema:
            schema_info = schema
        else:
            schema_info = "-- BigQuery DDL schema is not available or is empty. --"
    # Example for other database types (if ever needed)
    # elif callback_context.state.get("all_db_settings", {}).get("use_database") == "OtherDB":
    #     # Fetch and set schema for OtherDB
    #     # schema_info = get_other_db_schema_logic()
    #     schema_info = "-- Schema for OtherDB (logic to be implemented). --"

    base_instruction = return_instructions_root()
    # Ensure _invocation_context and agent are valid before modification
    if callback_context._invocation_context and callback_context._invocation_context.agent:
        callback_context._invocation_context.agent.instruction = (
            base_instruction
            + f"""

    --------- The schema of the relevant data with a few sample rows. ---------
    {schema_info}

    """
        )
    else:
        # Handle cases where context might not be fully initialized (e.g. log a warning)
        print("Warning: Agent invocation context not fully initialized in setup_before_agent_call.")


root_agent = Agent(
    model=os.getenv("ROOT_AGENT_MODEL"),
    name="db_ds_multiagent",
    instruction=return_instructions_root(),
    global_instruction=(
        f"""
        You are a Data Science and Data Analytics Multi Agent System.
        Todays date: {date_today}
        """
    ),
    sub_agents=[bqml_agent],
    tools=[
        call_db_agent,
        call_ds_agent,
        load_artifacts,
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
