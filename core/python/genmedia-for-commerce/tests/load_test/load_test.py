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

import json
import logging
import os
import time
import uuid

from locust import HttpUser, between, task

ENDPOINT = "/run_sse"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ChatStreamUser(HttpUser):
    """Simulates a user interacting with the chat stream API."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    @task
    def chat_stream(self) -> None:
        """Simulates a chat stream interaction."""
        headers = {"Content-Type": "application/json"}
        if os.environ.get("_ID_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['_ID_TOKEN']}"
        # Create session first
        user_id = f"user_{uuid.uuid4()}"
        session_data = {
            "state": {"preferred_language": "English", "visit_count": 1}
        }

        session_response = self.client.post(
            f"/apps/app/users/{user_id}/sessions",
            headers=headers,
            json=session_data,
        )

        # Get session_id from response
        session_id = session_response.json()["id"]

        # Send chat message
        data = {
            "app_name": "app",
            "user_id": user_id,
            "session_id": session_id,
            "new_message": {
                "role": "user",
                "parts": [{"text": "Hello! Weather in New york?"}],
            },
            "streaming": True,
        }
        start_time = time.time()

        with self.client.post(
            ENDPOINT,
            name=f"{ENDPOINT} message",
            headers=headers,
            json=data,
            catch_response=True,
            stream=True,
            params={"alt": "sse"},
        ) as response:
            if response.status_code == 200:
                events = []
                has_error = False
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        events.append(line_str)

                        if "429 Too Many Requests" in line_str:
                            self.environment.events.request.fire(
                                request_type="POST",
                                name=f"{ENDPOINT} rate_limited 429s",
                                response_time=0,
                                response_length=len(line),
                                response=response,
                                context={},
                            )

                        # Check for error responses in the JSON payload
                        try:
                            event_data = json.loads(line_str)
                            if (
                                isinstance(event_data, dict)
                                and "code" in event_data
                            ):
                                # Flag any non-2xx codes as errors
                                if event_data["code"] >= 400:
                                    has_error = True
                                    error_msg = event_data.get(
                                        "message", "Unknown error"
                                    )
                                    response.failure(
                                        f"Error in response: {error_msg}"
                                    )
                                    logger.error(
                                        "Received error response: code=%s, message=%s",
                                        event_data["code"],
                                        error_msg,
                                    )
                        except json.JSONDecodeError:
                            # If it's not valid JSON, continue processing
                            pass

                end_time = time.time()
                total_time = end_time - start_time

                # Only fire success event if no errors were found
                if not has_error:
                    self.environment.events.request.fire(
                        request_type="POST",
                        name=f"{ENDPOINT} end",
                        response_time=total_time
                        * 1000,  # Convert to milliseconds
                        response_length=len(events),
                        response=response,
                        context={},
                    )
            else:
                response.failure(
                    f"Unexpected status code: {response.status_code}"
                )
