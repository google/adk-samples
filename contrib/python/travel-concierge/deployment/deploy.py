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

"""Deployment script for Travel Concierge."""

import asyncio
import os

import vertexai
from absl import app, flags
from dotenv import load_dotenv
from google.adk.sessions import VertexAiSessionService
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

from travel_concierge.agent import root_agent

FLAGS = flags.FLAGS
flags.DEFINE_string("project_id", None, "GCP project ID.")
flags.DEFINE_string("location", None, "GCP location.")
flags.DEFINE_string("bucket", None, "GCP bucket.")

flags.DEFINE_string(
    "initial_states_path",
    None,
    "Relative path to the initial state file, .e.g eval/itinerary_empty_default.json",
)
flags.DEFINE_string("map_key", None, "API Key for Google Maps Grounding API")

flags.DEFINE_string("resource_id", None, "ReasoningEngine resource ID.")
flags.DEFINE_bool("create", False, "Creates a new deployment.")
flags.DEFINE_bool("quicktest", False, "Try a new deployment with one turn.")
flags.DEFINE_bool("delete", False, "Deletes an existing deployment.")
flags.mark_bool_flags_as_mutual_exclusive(["create", "delete", "quicktest"])


def create(env_vars: dict[str, str]) -> None:
    """Creates a new deployment."""
    print(env_vars)
    app = AdkApp(agent=root_agent, enable_tracing=True)

    remote_agent = agent_engines.create(
        app,
        display_name="Travel-Concierge-ADK",
        description="An Example AgentEngine Deployment",
        requirements=[
            "google-adk (>=1.31.0,<2.0.0)",
            "google-cloud-aiplatform[agent_engines] (>=1.93.1)",
            "google-genai (>=1.9.0)",
            "pydantic (>=2.10.6,<3.0.0)",
            "absl-py (>=2.2.1,<3.0.0)",
            "pydantic (>=2.10.6,<3.0.0)",
            "requests (>=2.32.3,<3.0.0)",
            "python-dotenv>=1.0.1",
            "arize-otel>=0.8.2; python_version >= '3.11' and python_version < '3.13'",
            "openinference-instrumentation-google-adk>=0.1.0; python_version >= '3.11' and python_version < '3.14'",
            "openinference-instrumentation>=0.1.34",
            "arize>=7.36.0",
            "arize-phoenix-evals>=0.20.8",
            "scikit-learn>=1.7.0",
            "pandas>=2.3.0",
        ],
        extra_packages=[
            "./travel_concierge",  # The main package
        ],
        env_vars=env_vars,
    )
    print(f"Created remote agent: {remote_agent.resource_name}")


def delete(resource_id: str) -> None:
    remote_agent = agent_engines.get(resource_id)
    remote_agent.delete(force=True)
    print(f"Deleted remote agent: {resource_id}")


def send_message(
    session_service: VertexAiSessionService, resource_id: str, message: str
) -> None:
    """Send a message to the deployed agent."""

    session = asyncio.run(
        session_service.create_session(
            app_name=resource_id, user_id="traveler0115"
        )
    )

    remote_agent = agent_engines.get(resource_id)

    print(f"Trying remote agent: {resource_id}")
    for event in remote_agent.stream_query(
        user_id="traveler0115",
        session_id=session.id,
        message=message,
    ):
        print(event)
    print("Done.")


def main(argv: list[str]) -> None:
    load_dotenv(override=True)
    env_vars = {}

    project_id = (
        FLAGS.project_id
        if FLAGS.project_id
        else os.getenv("GOOGLE_CLOUD_PROJECT")
    )

    location = (
        FLAGS.location if FLAGS.location else os.getenv("GOOGLE_CLOUD_LOCATION")
    )
    bucket = (
        FLAGS.bucket
        if FLAGS.bucket
        else os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")
    )
    # Variables for Travel Concierge from .env
    initial_states_path = (
        FLAGS.initial_states_path
        if FLAGS.initial_states_path
        else os.getenv("TRAVEL_CONCIERGE_SCENARIO")
    )
    env_vars["TRAVEL_CONCIERGE_SCENARIO"] = initial_states_path
    map_key = (
        FLAGS.initial_states_path
        if FLAGS.initial_states_path
        else os.getenv("GOOGLE_MAPS_API_KEY")
    )
    env_vars["GOOGLE_MAPS_API_KEY"] = map_key

    print(f"PROJECT: {project_id}")
    print(f"LOCATION: {location}")
    print(f"BUCKET: {bucket}")
    print(f"INITIAL_STATE: {initial_states_path}")
    print(f"MAP: {map_key[:5]}")

    if not project_id:
        print("Missing required environment variable: GOOGLE_CLOUD_PROJECT")
        return
    elif not location:
        print("Missing required environment variable: GOOGLE_CLOUD_LOCATION")
        return
    elif not bucket:
        print(
            "Missing required environment variable: GOOGLE_CLOUD_STORAGE_BUCKET"
        )
        return
    elif not initial_states_path:
        print(
            "Missing required environment variable: TRAVEL_CONCIERGE_SCENARIO"
        )
        return
    elif not map_key:
        print("Missing required environment variable: GOOGLE_MAPS_API_KEY")
        return

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket}",
    )

    if FLAGS.create:
        create(env_vars)
    elif FLAGS.delete:
        if not FLAGS.resource_id:
            print("resource_id is required for delete")
            return
        delete(FLAGS.resource_id)
    elif FLAGS.quicktest:
        if not FLAGS.resource_id:
            print("resource_id is required for quicktest")
            return
        session_service = VertexAiSessionService(project_id, location)
        send_message(
            session_service,
            FLAGS.resource_id,
            "Tell me more about activities I can do around Machu Picchu",
        )
    else:
        print("Unknown command")


if __name__ == "__main__":
    app.run(main)
