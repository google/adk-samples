# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A2A round trip, the path Gemini Enterprise uses to reach the agent.

Starts the server, checks the agent card has the v0.3 JSON-RPC shape Gemini
Enterprise registration needs, then sends a v0.3 `message/send` and checks the
workflow's answer comes back as a completed task. Calls Gemini, so it needs
credentials in .env; excluded from CI like the other integration tests:

    uv run pytest tests/integration/test_a2a_e2e.py
"""

import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator

import pytest
import requests
from requests.exceptions import RequestException


def _free_port() -> int:
    # Other local servers often hold 8000/8080; let the OS pick a free port.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


PORT = _free_port()
BASE_URL = f"http://127.0.0.1:{PORT}"
RPC_URL = f"{BASE_URL}/a2a/app"
CARD_URL = f"{RPC_URL}/.well-known/agent-card.json"


@pytest.fixture(scope="module")
def server() -> Iterator[None]:
    env = os.environ.copy()
    env.update(
        INTEGRATION_TEST="TRUE",
        APP_URL=BASE_URL,
        ADK_DISABLE_LOCAL_STORAGE="1",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.fast_api_app:app",
            "--port",
            str(PORT),
        ],
        env=env,
    )
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                if requests.get(CARD_URL, timeout=5).status_code == 200:
                    break
            except RequestException:
                pass
            time.sleep(1)
        else:
            pytest.fail("Server did not start")
        yield
    finally:
        process.terminate()
        process.wait()


def test_agent_card_has_v0_3_jsonrpc_interface(server: None) -> None:
    card = requests.get(CARD_URL, timeout=30).json()
    assert card["name"] and card["description"]
    assert card["skills"]
    interfaces = {
        (i["protocolBinding"], i["protocolVersion"]): i["url"]
        for i in card["supportedInterfaces"]
    }
    assert interfaces[("JSONRPC", "0.3")] == RPC_URL
    # Top-level fields of the 0.3 card shape.
    assert card["url"] == RPC_URL
    assert card["protocolVersion"] == "0.3"


def test_message_send_returns_completed_plan(server: None) -> None:
    body = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "parts": [
                    {
                        "kind": "text",
                        "text": "When can we do blade repairs at Elbe Marsh this week?",
                    }
                ],
            }
        },
    }
    response = requests.post(RPC_URL, json=body, timeout=300)
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["kind"] == "task"
    assert result["status"]["state"] == "completed"
    answer = " ".join(
        part.get("text", "")
        for artifact in result.get("artifacts", [])
        for part in artifact["parts"]
    )
    # The bundled sample's best blade-repair window at Elbe Marsh is 96.9%.
    assert "96.9" in answer
    assert "decision support" in answer.lower()


def test_agent_engine_adapter_creates_session(server: None) -> None:
    # Queries through the adapter need Vertex AI credentials (the Agent Engine
    # runtime forces GOOGLE_GENAI_USE_VERTEXAI=1); creating a session does not.
    response = requests.post(
        f"{BASE_URL}/api/reasoning_engine",
        json={
            "class_method": "async_create_session",
            "input": {"user_id": "u"},
        },
        timeout=60,
    )
    assert response.status_code == 200
    assert response.json()["output"]["id"]
