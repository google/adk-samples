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
"""A2A round-trip, the path Gemini Enterprise uses. Needs Gemini credentials.

Starts the FastAPI server, checks the agent card has the A2A 0.3 shape that
Gemini Enterprise registration validates, sends a JSON-RPC `message/send` and
checks the workflow's report comes back. Excluded from CI; run with:

    uv run pytest tests/integration/test_a2a.py
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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


PORT = _free_port()
BASE_URL = f"http://127.0.0.1:{PORT}"
RPC_URL = f"{BASE_URL}/a2a/app"
CARD_URL = f"{RPC_URL}/.well-known/agent-card.json"
# The workflow runs Bandit and the tests several times.
RUN_TIMEOUT_SECONDS = 900


@pytest.fixture(scope="module")
def server() -> Iterator[None]:
    env = {**os.environ, "INTEGRATION_TEST": "TRUE", "APP_URL": BASE_URL}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.fast_api_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                if requests.get(CARD_URL, timeout=5).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            pytest.fail("server did not start")
        yield
    finally:
        proc.terminate()
        proc.wait()


def test_agent_card_has_the_a2a_0_3_shape(server: None) -> None:
    card = requests.get(CARD_URL, timeout=10).json()
    assert card["name"] == "codemender_security_fixer"
    assert card["description"]
    assert card["url"] == RPC_URL
    assert card["protocolVersion"] == "0.3"
    assert card["preferredTransport"] == "JSONRPC"
    assert card["skills"]
    assert any(
        i["protocolBinding"] == "JSONRPC" and i["protocolVersion"] == "0.3"
        for i in card["supportedInterfaces"]
    )


def test_message_send_returns_the_report(server: None) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": uuid.uuid4().hex,
                "parts": [
                    {
                        "kind": "text",
                        "text": "Fix the security findings in the orders service.",
                    }
                ],
            }
        },
    }
    response = requests.post(
        RPC_URL,
        json=payload,
        headers={"A2A-Version": "0.3"},
        timeout=RUN_TIMEOUT_SECONDS,
    )
    body = response.json()
    assert "error" not in body, body.get("error")
    result = body["result"]
    assert result["status"]["state"] == "completed"
    text = "\n".join(
        part.get("text", "")
        for artifact in result.get("artifacts") or []
        for part in artifact["parts"]
    )
    assert "Security fix report" in text
    assert "| Bandit findings | 5 |" in text
    assert "```diff" in text
    assert "Nothing was committed" in text
