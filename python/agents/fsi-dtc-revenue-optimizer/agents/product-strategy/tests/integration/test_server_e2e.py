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
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import requests
from a2a.types import (
    JSONRPCErrorResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageResponse,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    TextPart,
)
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8000/"
A2A_RPC_URL = BASE_URL + "a2a/app/"
AGENT_CARD_URL = A2A_RPC_URL + ".well-known/agent-card.json"
FEEDBACK_URL = BASE_URL + "feedback"

HEADERS = {"Content-Type": "application/json"}


def log_output(pipe: Any, log_func: Any) -> None:
    """Log the output from the given pipe."""
    for line in iter(pipe.readline, ""):
        log_func(line.strip())


def start_server() -> subprocess.Popen[str]:
    """Start the FastAPI server using subprocess and log its output."""
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.fast_api_app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    env = os.environ.copy()
    env["INTEGRATION_TEST"] = "TRUE"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    # Start threads to log stdout and stderr in real-time
    threading.Thread(
        target=log_output, args=(process.stdout, logger.info), daemon=True
    ).start()
    threading.Thread(
        target=log_output, args=(process.stderr, logger.error), daemon=True
    ).start()

    return process


def wait_for_server(timeout: int = 90, interval: int = 1) -> bool:
    """Wait for the server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(AGENT_CARD_URL, timeout=10)
            if response.status_code == 200:
                logger.info("Server is ready")
                return True
        except RequestException:
            pass
        time.sleep(interval)
    logger.error(f"Server did not become ready within {timeout} seconds")
    return False


@pytest.fixture(scope="session")
def server_fixture(request: Any) -> Iterator[subprocess.Popen[str]]:
    """Pytest fixture to start and stop the server for testing."""
    logger.info("Starting server process")
    server_process = start_server()
    if not wait_for_server():
        pytest.fail("Server failed to start")
    logger.info("Server process started")

    def stop_server() -> None:
        logger.info("Stopping server process")
        server_process.terminate()
        server_process.wait()
        logger.info("Server process stopped")

    request.addfinalizer(stop_server)
    yield server_process


def test_chat_stream(server_fixture: subprocess.Popen[str]) -> None:
    """Test the chat stream functionality using A2A JSON-RPC protocol."""
    logger.info("Starting chat stream test")

    message = Message(
        message_id=f"msg-user-{uuid.uuid4()}",
        role=Role.user,
        parts=[Part(root=TextPart(text="Hi!"))],
    )

    request = SendStreamingMessageRequest(
        id="test-req-001",
        params=MessageSendParams(message=message),
    )

    # Send the request
    response = requests.post(
        A2A_RPC_URL,
        headers=HEADERS,
        json=request.model_dump(mode="json", exclude_none=True),
        stream=True,
        timeout=60,
    )
    assert response.status_code == 200

    # Parse streaming JSON-RPC responses
    responses: list[SendStreamingMessageResponse] = []

    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                event_json = line_str[6:]
                json_data = json.loads(event_json)
                streaming_response = SendStreamingMessageResponse.model_validate(
                    json_data
                )
                responses.append(streaming_response)

    assert responses, "No responses received from stream"

    # Check for final status update
    final_responses = [
        r.root
        for r in responses
        if hasattr(r.root, "result")
        and hasattr(r.root.result, "final")
        and r.root.result.final is True
    ]
    assert final_responses, "No final response received"

    final_response = final_responses[-1]
    assert final_response.result.kind == "status-update"
    assert hasattr(final_response.result, "status")
    assert final_response.result.status.state == "completed"

    # Check for artifact content
    artifact_responses = [
        r.root
        for r in responses
        if hasattr(r.root, "result") and r.root.result.kind == "artifact-update"
    ]
    assert artifact_responses, "No artifact content received in stream"

    # Verify text content is in the artifact
    artifact_response = artifact_responses[-1]
    assert hasattr(artifact_response.result, "artifact")
    artifact = artifact_response.result.artifact
    assert artifact.parts, "Artifact has no parts"

    has_text = any(
        part.root.kind == "text" and hasattr(part.root, "text") and part.root.text
        for part in artifact.parts
    )
    assert has_text, "No text content found in artifact"


def test_chat_non_streaming(server_fixture: subprocess.Popen[str]) -> None:
    """Test the non-streaming chat functionality using A2A JSON-RPC protocol."""
    logger.info("Starting non-streaming chat test")

    message = Message(
        message_id=f"msg-user-{uuid.uuid4()}",
        role=Role.user,
        parts=[Part(root=TextPart(text="Hi!"))],
    )

    request = SendMessageRequest(
        id="test-req-002",
        params=MessageSendParams(message=message),
    )

    response = requests.post(
        A2A_RPC_URL,
        headers=HEADERS,
        json=request.model_dump(mode="json", exclude_none=True),
        timeout=60,
    )
    assert response.status_code == 200

    # Parse the single JSON-RPC response
    response_data = response.json()
    message_response = SendMessageResponse.model_validate(response_data)
    logger.info(f"Received response: {message_response}")

    # For non-streaming, the result is a Task object
    json_rpc_resp = message_response.root
    assert hasattr(json_rpc_resp, "result")
    task = json_rpc_resp.result
    assert task.kind == "task"
    assert hasattr(task, "status")
    assert task.status.state == "completed"

    # Check that we got artifacts (the final agent output)
    assert hasattr(task, "artifacts")
    assert task.artifacts, "No artifacts in task"

    # Verify we got text content in the artifact
    artifact = task.artifacts[0]
    assert artifact.parts, "Artifact has no parts"

    has_text = any(
        part.root.kind == "text" and hasattr(part.root, "text") and part.root.text
        for part in artifact.parts
    )
    assert has_text, "No text content found in artifact"


def test_chat_stream_error_handling(server_fixture: subprocess.Popen[str]) -> None:
    """Test the chat stream error handling with invalid A2A request."""
    logger.info("Starting chat stream error handling test")

    invalid_data = {
        "jsonrpc": "2.0",
        "id": "test-error-001",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                # Missing required 'parts' field
                "messageId": f"msg-user-{uuid.uuid4()}",
            }
        },
    }

    response = requests.post(
        A2A_RPC_URL, headers=HEADERS, json=invalid_data, timeout=10
    )
    assert response.status_code == 200

    response_data = response.json()
    error_response = JSONRPCErrorResponse.model_validate(response_data)
    assert "error" in response_data, "Expected JSON-RPC error in response"

    # Assert error for invalid parameters
    assert error_response.error.code == -32602

    logger.info("Error handling test completed successfully")


def test_collect_feedback(server_fixture: subprocess.Popen[str]) -> None:
    """
    Test the feedback collection endpoint (/feedback) to ensure it properly
    logs the received feedback.
    """
    # Create sample feedback data
    feedback_data = {
        "score": 4,
        "user_id": "test-user-456",
        "session_id": "test-session-456",
        "text": "Great response!",
    }

    response = requests.post(
        FEEDBACK_URL, json=feedback_data, headers=HEADERS, timeout=10
    )
    assert response.status_code == 200


def test_a2a_agent_json_generation(server_fixture: subprocess.Popen[str]) -> None:
    """
    Test that the agent.json file is automatically generated and served correctly
    via the well-known URI.
    """
    # Verify the A2A endpoint serves the agent card
    response = requests.get(AGENT_CARD_URL, timeout=10)
    assert response.status_code == 200, f"A2A endpoint returned {response.status_code}"

    # Validate required fields in served agent card
    served_agent_card = response.json()
    required_fields = [
        "name",
        "description",
        "skills",
        "capabilities",
        "url",
        "version",
    ]
    for field in required_fields:
        assert field in served_agent_card, (
            f"Missing required field in served agent card: {field}"
        )
