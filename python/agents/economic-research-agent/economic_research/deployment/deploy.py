# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""
Modernized Deployment script for Economic Research Agent using Vertex AI Agent Engine (ADK 2.1+).
"""

import logging
import os

import cloudpickle
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp
import json
import datetime
from typing import Any
from dotenv import set_key

import economic_research
from economic_research.agent import ERAAgent

logging.getLogger("google.cloud.aiplatform").setLevel(logging.DEBUG)
cloudpickle.register_pickle_by_value(economic_research)

def update_env_file(agent_engine_id: str, env_file_path: str):
    """Updates the .env file with the agent engine ID."""
    try:
        set_key(env_file_path, "AGENT_ENGINE_ID", agent_engine_id)
        print(f"Updated AGENT_ENGINE_ID in {env_file_path} to {agent_engine_id}")
    except Exception as e:
        print(f"Error updating .env file: {e}")

def write_deployment_metadata(
    remote_agent: Any,
    metadata_file: str = "deployment_metadata.json",
) -> None:
    """Write deployment metadata to file."""
    metadata = {
        "remote_agent_engine_id": remote_agent.resource_name,
        "deployment_target": "agent_engine",
        "is_a2a": False,
        "deployment_timestamp": datetime.datetime.now().isoformat(),
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Agent Engine ID written to {metadata_file}")

def deploy_era_to_vertex(project_id: str, location: str = "us-east1"):
    print(
        f"🚀 Initializing Modern Agent Engine Deployment for economic-research in {location}..."
    )

    # Defaulting to standard naming pattern for staging buckets
    staging_bucket = os.getenv(
        "GOOGLE_CLOUD_STORAGE_BUCKET", f"gs://{project_id}-agent-engine-v16"
    )
    print(f"🪣 Using staging bucket: {staging_bucket}")

    vertexai.init(
        project=project_id, location=location, staging_bucket=staging_bucket
    )

    # Calculate absolute path for extra_packages
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(package_dir)
    agent_package_path = package_dir

    print(f"📦 Packaging from: {agent_package_path}")

    # Read requirements from requirements.txt
    requirements_path = os.path.join(project_root, "requirements.txt")
    with open(requirements_path) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    era_instance = ERAAgent()
    root_agent = era_instance.get_app().root_agent

    adk_app = AdkApp(
        agent=root_agent,
        enable_tracing=False,
    )

    # Use the new agent_engines API
    remote_agent = agent_engines.create(
        adk_app,
        requirements=requirements,
        extra_packages=[os.path.join(project_root, "economic_research")],
        display_name="adk-economic-agent",
    )

    print("✅ Modern Deployment Successful!")
    print(f"Agent Engine ID: {remote_agent.resource_name}")
    
    # Apply fix for automatic ID picking
    env_file_path = os.path.join(project_root, ".env")
    update_env_file(remote_agent.resource_name, env_file_path)
    write_deployment_metadata(remote_agent, os.path.join(project_root, "deployment_metadata.json"))
    
    return remote_agent.resource_name


if __name__ == "__main__":
    import google.auth

    try:
        _, project = google.auth.default()
        active_project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not active_project:
            raise ValueError(
                "Active GCP project could not be determined. Please set GOOGLE_CLOUD_PROJECT."
            )
        deploy_era_to_vertex(project_id=active_project)
    except Exception as e:
        print(f"❌ Modern Deployment Failed: {e}")
