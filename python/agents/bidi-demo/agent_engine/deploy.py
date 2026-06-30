"""
Deploy the bidi-demo agent to Vertex AI Agent Engine.

Requires the agent-engine optional dependencies:
    uv sync --extra agent-engine

Usage:
    uv run agent_engine/deploy.py
"""

import os
import sys

import vertexai
from dotenv import load_dotenv
from vertexai import types as vertexai_types
from vertexai.preview.reasoning_engines import AdkApp

load_dotenv(
    os.path.join(os.path.dirname(__file__), "..", "app", ".env"), override=True
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1")
STAGING_BUCKET = os.environ["STAGING_BUCKET"]


def main():
    from google_search_agent.agent import agent

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

    print(f"Project:  {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Bucket:   {STAGING_BUCKET}")
    print(f"Model:    {agent.model}")

    # Ensure staging bucket exists
    print("\nCreating staging bucket (ignored if exists)...")
    os.system(
        f"gsutil mb -p {PROJECT_ID} -l {LOCATION} {STAGING_BUCKET} 2>/dev/null || true"
    )

    def session_service_builder():
        from google.adk.sessions.in_memory_session_service import (
            InMemorySessionService,
        )

        return InMemorySessionService()

    def memory_service_builder():
        from google.adk.memory.in_memory_memory_service import (
            InMemoryMemoryService,
        )

        return InMemoryMemoryService()

    app = AdkApp(
        agent=agent,
        session_service_builder=session_service_builder,
        memory_service_builder=memory_service_builder,
    )

    # Deploy with EXPERIMENTAL mode for bidi streaming
    print("\nDeploying agent to Agent Engine (this takes a few minutes)...")
    remote_agent = client.agent_engines.create(
        agent=app,
        config={
            "display_name": "ADK Bidi Demo (Google Search)",
            "description": "ADK Gemini Live API Toolkit demo with Google Search",
            "requirements": [
                "google-cloud-aiplatform[agent_engines,adk]>=1.112",
                "google-adk",
                "google-genai",
                "cloudpickle",
                "pydantic",
            ],
            "staging_bucket": STAGING_BUCKET,
            "agent_server_mode": vertexai_types.AgentServerMode.EXPERIMENTAL,
        },
    )

    resource_name = remote_agent.api_resource.name
    print("\nAgent deployed successfully!")
    print(f"Resource name: {resource_name}")

    with open("agent_resource_name.txt", "w") as f:
        f.write(resource_name)
    print("Resource name saved to agent_resource_name.txt")


if __name__ == "__main__":
    main()
