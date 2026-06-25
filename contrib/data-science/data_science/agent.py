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

import base64
import json
import logging
import os
from datetime import date

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext

# from google.adk.tools import load_artifacts
from google.genai import types
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from .prompts import return_instructions_root
from .sub_agents import bqml_agent
from .sub_agents.alloydb.tools import (
    get_database_settings as get_alloydb_database_settings,
)
from .sub_agents.bigquery.tools import (
    get_database_settings as get_bq_database_settings,
)
from .tools import call_alloydb_agent, call_analytics_agent, call_bigquery_agent

# Configure Weave endpoint and authentication.
_WANDB_BASE_URL = "https://trace.wandb.ai"
_WANDB_PROJECT_ID = os.getenv("WANDB_PROJECT_ID")
_OTEL_EXPORTER_OTLP_ENDPOINT = f"{_WANDB_BASE_URL}/otel/v1/traces"

# Set up authentication
_WANDB_API_KEY = os.getenv("WANDB_API_KEY")
_WANDB_AUTH = base64.b64encode(f"api:{_WANDB_API_KEY}".encode()).decode()

_OTEL_EXPORTER_OTLP_HEADERS = {
    "Authorization": f"Basic {_WANDB_AUTH}",
    "project_id": _WANDB_PROJECT_ID,
}

# Create the OTLP span exporter with endpoint and headers
exporter = OTLPSpanExporter(
    endpoint=_OTEL_EXPORTER_OTLP_ENDPOINT,
    headers=_OTEL_EXPORTER_OTLP_HEADERS,
)

# Create a tracer provider and add the exporter
_tracer_provider = trace_sdk.TracerProvider()
_tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

# Set the global tracer provider BEFORE importing/using ADK
trace.set_tracer_provider(_tracer_provider)

# Set up logging
# Note this level can be overridden by adk web on the command line;
# e.g. running `adk web --log_level DEBUG` or `adk web -v`
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Initialize module-level config variables
_dataset_config = {}
_database_settings = {}
_supported_dataset_types = ["bigquery", "alloydb"]
_required_dataset_config_params = ["name", "description"]


def load_dataset_config():
    """Load the dataset configurations for the agent from the config file"""

    dataset_config_file = os.getenv("DATASET_CONFIG_FILE", "")
    if not dataset_config_file:
        _logger.fatal("DATASET_CONFIG_FILE env var not set")

    with open(dataset_config_file, encoding="utf-8") as f:
        dataset_config = json.load(f)

    if "datasets" not in dataset_config:
        _logger.fatal("No 'datasets' entry in dataset config")

    for dataset in dataset_config["datasets"]:
        if "type" not in dataset:
            _logger.fatal("Missing dataset type")
        if dataset["type"] not in _supported_dataset_types:
            _logger.fatal("Dataset type '%s' not supported", dataset["type"])

        for p in _required_dataset_config_params:
            if p not in dataset:
                _logger.fatal(
                    "Missing required param '%s' from %s dataset config",
                    p,
                    dataset["type"],
                )

    return dataset_config


def get_database_settings(db_type: str) -> dict:
    """Wrapper function to get database settings by type"""
    assert db_type in _supported_dataset_types
    if db_type == "bigquery":
        return get_bq_database_settings()
    else:
        return get_alloydb_database_settings()


def init_database_settings(dataset_config: dict) -> dict:
    """Initializes the database settings for the configured datasets"""
    db_settings = {}
    for dataset in dataset_config["datasets"]:
        db_settings[dataset["type"]] = get_database_settings(dataset["type"])
    return db_settings


def get_dataset_definitions_for_instructions() -> str:
    """Returns the dataset definitions instructions block"""

    dataset_definitions = """
<DATASETS>
"""
    for dataset in _dataset_config["datasets"]:
        dataset_type = dataset["type"]
        dataset_definitions += f"""
<{dataset_type.upper()}>
<DESCRIPTION>
{dataset["description"]}
</DESCRIPTION>
<SCHEMA>
--------- The schema of the relevant database with a few sample rows. --------
{_database_settings[dataset_type]["schema"]}
</SCHEMA>
</{dataset_type.upper()}>

"""
    dataset_definitions += """
</DATASETS>
"""

    if "cross_dataset_relations" in _dataset_config:
        dataset_definitions += f"""
<CROSS_DATASET_RELATIONS>
--------- The cross dataset relations between the configured datasets. ---------
{_dataset_config["cross_dataset_relations"]}
</CROSS_DATASET_RELATIONS>
"""

    return dataset_definitions


def load_database_settings_in_context(callback_context: CallbackContext):
    """Load database settings into the callback context on first use."""
    if "database_settings" not in callback_context.state:
        callback_context.state["database_settings"] = _database_settings


def get_root_agent() -> LlmAgent:
    tools = [call_analytics_agent]
    sub_agents = []
    for dataset in _dataset_config["datasets"]:
        if dataset["type"] == "bigquery":
            tools.append(call_bigquery_agent)
            sub_agents.append(bqml_agent)
        elif dataset["type"] == "alloydb":
            tools.append(call_alloydb_agent)

    agent = LlmAgent(
        model=os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash"),
        name="data_science_root_agent",
        instruction=return_instructions_root()
        + get_dataset_definitions_for_instructions(),
        global_instruction=(
            f"""
            You are a Data Science and Data Analytics Multi Agent System.
            Todays date: {date.today()}
            """
        ),
        sub_agents=sub_agents,  # type: ignore
        tools=tools,  # type: ignore
        before_agent_callback=load_database_settings_in_context,
        generate_content_config=types.GenerateContentConfig(temperature=0.01),
    )

    return agent


# Initialize dataset configurations and database info before the agent starts
_dataset_config = load_dataset_config()
_database_settings = init_database_settings(_dataset_config)


# Fetch the root agent
root_agent = get_root_agent()
