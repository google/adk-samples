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
    AGENT_ENGINE_ID = os.environ["AGENT_ENGINE_ID"]

    if not PROJECT:
        print("Missing required environment variable: GOOGLE_CLOUD_PROJECT")
        return
    elif not LOCATION:
        print("Missing required environment variable: GOOGLE_CLOUD_LOCATION")
        return
    elif not STAGING_BUCKET:
        print("Missing required environment variable: GOOGLE_CLOUD_STORAGE_BUCKET")
        return
    elif not AGENT_ENGINE_ID:
        print("Missing required environment variable: AGENT_ENGINE_ID")
        return

    print(f"PROJECT: {PROJECT}")
    print(f"LOCATION: {LOCATION}")
    print(f"STAGING_BUCKET: {STAGING_BUCKET}")
    print(f"AGENT_ENGINE_ID: {AGENT_ENGINE_ID}")

    vertexai.init(
        project=PROJECT,
        location=LOCATION,
        staging_bucket=f"gs://{STAGING_BUCKET}",
    )

    user_id="user"
    agent = agent_engines.get(AGENT_ENGINE_ID)
    session = agent.create_session(user_id=user_id)
    print("Type 'quit' to exit.")
    while True:
        user_input = input("Input: ")
        if user_input == "quit":
            break

        for event in agent.stream_query(
            user_id=user_id, session_id=session["id"], message=user_input
        ):
            if "content" in event:
                if "parts" in event["content"]:
                    parts = event["content"]["parts"]
                    for part in parts:
                        if "text" in part:
                            text_part = part["text"]
                            print(f"Response: {text_part}")

    agent.delete_session(user_id=user_id, session_id=session["id"])

if __name__ == "__main__":
    app.run(main)