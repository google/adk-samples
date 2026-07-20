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

"""Deploy, test, or delete the Document Generation Agent on Vertex AI Agent Engine.

Run from the agent's root directory (intelligent-document-generation-agent/).

Usage:
    python deployment/deploy.py --deploy
    python deployment/deploy.py --deploy --test
    python deployment/deploy.py --test --resource_id <full-resource-name>
    python deployment/deploy.py --delete --resource_id <full-resource-name>
"""

import argparse
import asyncio
import logging
import sys

import vertexai
from google.api_core.exceptions import GoogleAPIError, NotFound
from vertexai import agent_engines

from intelligent_document_generation_agent.agent import root_agent
from intelligent_document_generation_agent.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_QUERY = "Acknowledge receipt and briefly describe what you do."


def _client() -> vertexai.Client:
    return vertexai.Client(
        project=settings.GOOGLE_CLOUD_PROJECT,
        location="global",
    )


def deploy_agent(display_name: str) -> "agent_engines.AgentEngine":
    """Deploy the agent to Vertex AI Agent Engine."""
    logger.info(f"Deploying app with display name: {display_name}...")

    # Agent Engine sets GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION at runtime;
    # also skip empty values and the runtime-assigned reasoning engine id.
    deploy_env_vars = {
        key: value
        for key, value in settings.model_dump().items()
        if value is not None
        and key
        not in {"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "REASONING_ENGINE"}
    }
    logger.info(f"Passing env vars to deployment: {list(deploy_env_vars.keys())}")

    try:
        remote_app = _client().agent_engines.create(
            agent=agent_engines.AdkApp(agent=root_agent),
            config={
                "display_name": display_name,
                "requirements": "intelligent_document_generation_agent/requirements.txt",
                "extra_packages": ["intelligent_document_generation_agent"],
                "staging_bucket": f"gs://{settings.ADK_STAGING_BUCKET}",
                "env_vars": deploy_env_vars,
                "service_account": settings.PROJECT_SERVICE_ACCOUNT,
            },
        )
        logger.info(f"Agent deployed successfully: {remote_app.resource_name}")
        return remote_app
    except GoogleAPIError as e:
        logger.error(f"Failed to deploy agent: {e}")
        sys.exit(1)


async def test_agent_deployment(remote_app: "agent_engines.AgentEngine") -> None:
    """Smoke-test a deployed agent by streaming a single query."""
    logger.info("Testing deployment...")
    try:
        session = remote_app.create_session(user_id="123")
        logger.info(f"Session created with ID: {session['id']}")
        print("\n--- Agent Response ---")
        for event in remote_app.stream_query(
            user_id="123",
            session_id=session["id"],
            message=TEST_QUERY,
        ):
            if event.get("content"):
                print(event["content"])
        print("--- End Agent Response ---\n")
        logger.info("Agent test query completed successfully.")
    except GoogleAPIError as e:
        logger.error(f"Failed to test agent: {e}")
        sys.exit(1)


def delete_agent(resource_id: str) -> None:
    """Delete a deployed agent by its full resource name."""
    logger.info(f"Attempting to delete agent: {resource_id}")
    client = _client()
    try:
        client.agent_engines.get(name=resource_id)
        client.agent_engines.delete(name=resource_id, force=True)
        logger.info(f"Agent {resource_id} deleted successfully.")
    except NotFound:
        logger.warning(f"Agent {resource_id} not found. Nothing to delete.")
    except GoogleAPIError as e:
        logger.error(f"Failed to delete agent {resource_id}: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the Document Generation Agent on Vertex AI Agent Engine."
    )
    parser.add_argument("--deploy", action="store_true", help="Deploy the agent.")
    parser.add_argument(
        "--test", action="store_true", help="Send a test query to a deployed agent."
    )
    parser.add_argument(
        "--delete", action="store_true", help="Delete a deployed agent."
    )
    parser.add_argument(
        "--resource_id",
        help="Full resource name: projects/<project>/locations/<location>/reasoningEngines/<id>. Required with --delete, or with --test when not deploying in the same run.",
    )
    parser.add_argument(
        "--display_name",
        default="Intelligent Document Generation Agent",
        help="Display name for the deployed agent.",
    )

    args = parser.parse_args()

    if args.delete:
        if not args.resource_id:
            parser.error("--resource_id is required when using --delete.")
        delete_agent(args.resource_id)
        return

    if args.deploy:
        remote_app = deploy_agent(args.display_name)
        if args.test:
            asyncio.run(test_agent_deployment(remote_app))
        return

    if args.test:
        if not args.resource_id:
            parser.error(
                "--resource_id is required when using --test without --deploy."
            )
        try:
            existing = _client().agent_engines.get(name=args.resource_id)
        except NotFound:
            logger.error(f"Agent with resource ID '{args.resource_id}' not found.")
            sys.exit(1)
        except GoogleAPIError as e:
            logger.error(f"Error retrieving agent {args.resource_id}: {e}")
            sys.exit(1)
        asyncio.run(test_agent_deployment(existing))
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
