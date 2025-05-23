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

import argparse
import logging
import sys

import vertexai
from customer_service.agent import root_agent
from customer_service.config import Config
from google.api_core.exceptions import NotFound
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

configs = Config()

STAGING_BUCKET = f"gs://{configs.CLOUD_PROJECT}-adk-customer-service-staging"

AGENT_WHL_FILE = "./customer_service-0.1.0-py3-none-any.whl"

vertexai.init(
    project=configs.CLOUD_PROJECT,
    location=configs.CLOUD_LOCATION,
    staging_bucket=STAGING_BUCKET,
)

parser = argparse.ArgumentParser(description="Short sample app")

parser.add_argument(
    "--delete",
    action="store_true",
    dest="delete",
    required=False,
    help="Delete deployed agent",
)
parser.add_argument(
    "--resource_id",
    required="--delete" in sys.argv,
    action="store",
    dest="resource_id",
    help="The resource id of the agent to be deleted in the format projects/PROJECT_ID/locations/LOCATION/reasoningEngines/REASONING_ENGINE_ID",
)


args = parser.parse_args()

if args.delete:
    try:
        agent_engines.get(resource_name=args.resource_id)
        agent_engines.delete(resource_name=args.resource_id)
        print(f"Agent {args.resource_id} deleted successfully")
    except NotFound as e:
        print(e)
        print(f"Agent {args.resource_id} not found")

else:
    logger.info("deploying app...")
    app = AdkApp(agent=root_agent, enable_tracing=False)

    logging.debug("deploying agent to agent engine:")
    remote_app = agent_engines.create(
        app,
        requirements=[
            AGENT_WHL_FILE,
        ],
        extra_packages=[AGENT_WHL_FILE],
    )

    logging.debug("testing deployment:")
    session = remote_app.create_session(user_id="123")
    for event in remote_app.stream_query(
        user_id="123",
        session_id=session["id"],
        message="hello!",
    ):
        if event.get("content", None):
            print(
                f"Agent deployed successfully under resource name: {remote_app.resource_name}"
            )
