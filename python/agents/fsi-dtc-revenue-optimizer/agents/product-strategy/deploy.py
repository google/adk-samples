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

"""Custom deploy script for Agent Engine with GcsArtifactService.

The `adk deploy agent_engine` CLI does not support artifact_service_builder,
so we use the AdkApp API directly to pass GcsArtifactService as the
artifact_service_builder.

Usage:
    uv run python deploy.py --project=<project> --region=<region> \
        --display_name=<name> [--agent_engine_id=<id>]
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

import vertexai
from google.adk.artifacts import GcsArtifactService
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp


def get_artifact_service():
    """Creates a GcsArtifactService using the LOGS_BUCKET_NAME env var."""
    bucket_name = os.environ.get("LOGS_BUCKET_NAME")
    if not bucket_name:
        raise ValueError(
            "LOGS_BUCKET_NAME environment variable is required for "
            "GcsArtifactService. Set it in .env."
        )
    service = GcsArtifactService(bucket_name=bucket_name)
    print(f"Created GcsArtifactService for bucket: {bucket_name}")
    return service


def main():
    parser = argparse.ArgumentParser(
        description="Deploy product-strategy agent to Agent Engine"
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project ID",
    )
    parser.add_argument(
        "--region",
        default="us-central1",
        help="Agent Engine region (default: us-central1)",
    )
    parser.add_argument(
        "--display_name",
        default="product-strategy",
        help="Display name for the agent",
    )
    parser.add_argument(
        "--agent_engine_id",
        default=None,
        help="Existing Agent Engine ID to update (full resource name or ID)",
    )
    args = parser.parse_args()

    if not args.project:
        print("Error: --project or GOOGLE_CLOUD_PROJECT env var required.")
        sys.exit(1)

    bucket_name = os.environ.get("LOGS_BUCKET_NAME")
    if not bucket_name:
        print("Error: LOGS_BUCKET_NAME env var required.")
        sys.exit(1)

    # Import the agent
    from app.agent import root_agent

    # Build env vars to pass to Agent Engine (exclude ones set by the platform)
    env_vars = {}
    for key in [
        "ADK_MODEL",
        "VEO_GCS_BUCKET",
        "LOGS_BUCKET_NAME",
        "GOOGLE_CLOUD_LOCATION",
    ]:
        val = os.environ.get(key)
        if val:
            env_vars[key] = val

    # Initialize Vertex AI
    staging_bucket = f"gs://{bucket_name}"
    vertexai.init(
        project=args.project,
        location=args.region,
        staging_bucket=staging_bucket,
    )
    print(f"Vertex AI initialized: project={args.project}, region={args.region}")

    # Create AdkApp with GcsArtifactService
    adk_app = AdkApp(
        agent=root_agent,
        artifact_service_builder=get_artifact_service,
        env_vars=env_vars,
    )
    adk_app.set_up()
    print("AdkApp created with GcsArtifactService")

    # Read requirements from requirements.txt
    requirements = []
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        with open(req_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)

    # The key fix for ModuleNotFoundError: pass the app directory directly
    extra_packages = ["./app"]

    if args.agent_engine_id:
        # Update existing agent
        print(f"Updating agent: {args.agent_engine_id}")
        existing = agent_engines.AgentEngine(args.agent_engine_id)
        remote_agent = existing.update(
            agent_engine=adk_app,
            requirements=requirements,
            extra_packages=extra_packages,
            env_vars=env_vars,
            display_name=args.display_name,
        )
        print(f"✅ Updated agent engine: {remote_agent.resource_name}")
    else:
        # Create new agent
        print("Creating new agent...")
        remote_agent = agent_engines.create(
            adk_app,
            requirements=requirements,
            extra_packages=extra_packages,
            env_vars=env_vars,
            display_name=args.display_name,
        )
        print(f"✅ Created agent engine: {remote_agent.resource_name}")


if __name__ == "__main__":
    main()
