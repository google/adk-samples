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

from absl import app
import os
from dotenv import load_dotenv
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp
from auto_insurance_agent.agent import root_agent

def main(argv: list[str]) -> None:

    load_dotenv()

    PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
    LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]
    STAGING_BUCKET = os.environ["GOOGLE_CLOUD_STORAGE_BUCKET"]

    if not PROJECT:
        print("Missing required environment variable: GOOGLE_CLOUD_PROJECT")
        return
    elif not LOCATION:
        print("Missing required environment variable: GOOGLE_CLOUD_LOCATION")
        return
    elif not STAGING_BUCKET:
        print("Missing required environment variable: GOOGLE_CLOUD_STORAGE_BUCKET")
        return

    print(f"PROJECT: {PROJECT}")
    print(f"LOCATION: {LOCATION}")
    print(f"STAGING_BUCKET: {STAGING_BUCKET}")

    vertexai.init(
        project=PROJECT,
        location=LOCATION,
        staging_bucket=f"gs://{STAGING_BUCKET}",
    )

    app = AdkApp(agent=root_agent, enable_tracing=False)

    remote_agent = agent_engines.create(
        app,
        requirements=[
            "google-adk>=1.0.0,<2.0.0",
            "google-cloud-aiplatform[adk,agent_engines]",
            "google-cloud-secret-manager"
        ]
    )

    print(f"Created remote agent: {remote_agent.resource_name}")
    print(f"Before testing, run the following: export AGENT_ENGINE_ID={remote_agent.resource_name}")

if __name__ == "__main__":
    app.run(main)
