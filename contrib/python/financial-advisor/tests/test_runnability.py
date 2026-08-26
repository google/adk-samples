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
"""Runnability tests for the recipe."""

import os
from unittest.mock import MagicMock, patch


def test_agent_runnability() -> None:
    """Verify agent.py imports and defines the expected globals."""
    # provide a dummy GCP project and patch google.auth.default() so import-time
    # credential lookups don't need ADC — the setup must happen before the import.
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

    with patch(
        "google.auth.default", return_value=(MagicMock(), "test-project")
    ):
        import financial_advisor.agent

    assert financial_advisor.agent.root_agent is not None
    tools = financial_advisor.agent.root_agent.tools
    assert len(tools) == 4
    tool_names = [t.name for t in tools]
    assert "data_analyst_agent" in tool_names
    assert "trading_analyst_agent" in tool_names
    assert "execution_analyst_agent" in tool_names
    assert "risk_analyst_agent" in tool_names
    for tool in tools:
        assert tool.description, f"Tool {tool.name} missing description"


def test_a2a_protocol_and_api_surface() -> None:
    """Verify A2A agent card generation, endpoint routing, and error resilience."""
    import asyncio

    from fastapi.testclient import TestClient

    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

    with patch(
        "google.auth.default", return_value=(MagicMock(), "test-project")
    ):
        from financial_advisor.agent import root_agent
        from financial_advisor.app_utils.a2a import generate_agent_card
        from financial_advisor.fast_api_app import app

    # 1. Test generate_agent_card directly
    card = asyncio.run(
        generate_agent_card(
            agent=root_agent,
            rpc_url="http://0.0.0.0:8080/a2a/financial_advisor",
        )
    )
    assert card.name == "financial_coordinator"
    assert card.capabilities.streaming is True
    assert len(card.skills) >= 5
    versions = [iface.protocol_version for iface in card.supported_interfaces]
    assert "1.0" in versions
    assert "0.3" in versions
    assert versions.count("0.3") == 1

    with TestClient(app) as client:
        # 2. Test root redirect
        resp_redirect = client.get(
            "/.well-known/agent-card.json", follow_redirects=False
        )
        assert resp_redirect.status_code == 307
        assert (
            resp_redirect.headers.get("location")
            == "/a2a/financial_advisor/.well-known/agent-card.json"
        )

        # 3. Test card endpoint idempotency across repeated queries
        for _ in range(3):
            resp_card = client.get(
                "/a2a/financial_advisor/.well-known/agent-card.json"
            )
            assert resp_card.status_code == 200
            card_json = resp_card.json()
            ifaces = card_json.get("supportedInterfaces", [])
            card_versions = [x.get("protocolVersion") for x in ifaces]
            assert card_versions.count("0.3") == 1
            assert card_versions.count("1.0") == 1

        # 4. Test malformed JSON-RPC requests return structured error responses
        # Missing jsonrpc version
        bad_rpc = client.post(
            "/a2a/financial_advisor",
            json={"id": "test-err-1", "method": "message/send"},
        )
        assert bad_rpc.status_code == 200
        assert "error" in bad_rpc.json()
        assert bad_rpc.json()["error"]["code"] == -32600

        # Method not found
        bad_method = client.post(
            "/a2a/financial_advisor",
            json={
                "jsonrpc": "2.0",
                "id": "test-err-2",
                "method": "non_existent_method",
            },
        )
        assert bad_method.status_code == 200
        assert "error" in bad_method.json()
        assert bad_method.json()["error"]["code"] == -32601

        # 5. Test healthz probe
        resp_healthz = client.get("/healthz")
        assert resp_healthz.status_code == 200
        assert resp_healthz.json() == {"status": "ok"}

        # 6. Test reasoning engine malformed body validation
        bad_re = client.post("/api/reasoning_engine", json={})
        assert bad_re.status_code == 400
        assert "class_method" in bad_re.json()["detail"]

        # 7. Test reasoning engine TypeError on invalid kwargs returns 400 not 500
        bad_args = client.post(
            "/api/reasoning_engine",
            json={
                "class_method": "create_session",
                "input": {"unsupported_kwarg_xyz": 123},
            },
        )
        assert bad_args.status_code == 400
        assert "Invalid arguments" in bad_args.json()["detail"]


def test_security_and_governance_posture() -> None:
    """Verify security controls, prompt boundaries, and container hermeticity."""
    recipe_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Container hermeticity: check .dockerignore exclusions
    dockerignore_path = os.path.join(recipe_dir, ".dockerignore")
    assert os.path.exists(dockerignore_path)
    with open(dockerignore_path, encoding="utf-8") as f:
        dockerignore_content = f.read()
    assert ".env*" in dockerignore_content
    assert "!.env.example" in dockerignore_content
    assert "*.pem" in dockerignore_content
    assert "*.key" in dockerignore_content
    assert ".idea/" in dockerignore_content
    assert ".vscode/" in dockerignore_content
    assert ".adk/" in dockerignore_content

    # 2. Supply chain & secret hygiene: pyproject.toml does not leak employee email
    pyproject_path = os.path.join(recipe_dir, "pyproject.toml")
    with open(pyproject_path, encoding="utf-8") as f:
        pyproject_content = f.read()
    assert "@google.com" not in pyproject_content

    # 3. OWASP Top 10 for LLMs: Prompt injection defenses & zero execution agency
    from financial_advisor.prompt import FINANCIAL_COORDINATOR_PROMPT
    from financial_advisor.sub_agents.data_analyst.prompt import (
        DATA_ANALYST_PROMPT,
    )

    assert "Role Integrity & Anti-Jailbreak" in FINANCIAL_COORDINATOR_PROMPT
    assert "Zero Execution Agency" in FINANCIAL_COORDINATOR_PROMPT
    assert "Indirect Prompt Injection Defense" in DATA_ANALYST_PROMPT

    # 4. Excessive agency: verify agent tools are strictly bounded and read-only
    with patch(
        "google.auth.default", return_value=(MagicMock(), "test-project")
    ):
        from google.adk.tools import google_search

        from financial_advisor.agent import root_agent
        from financial_advisor.sub_agents.data_analyst import data_analyst_agent
        from financial_advisor.sub_agents.execution_analyst import (
            execution_analyst_agent,
        )
        from financial_advisor.sub_agents.risk_analyst import risk_analyst_agent
        from financial_advisor.sub_agents.trading_analyst import (
            trading_analyst_agent,
        )

    # Root agent has only subagent tools
    assert len(root_agent.tools) == 4
    # Data analyst has only google_search
    assert data_analyst_agent.tools == [google_search]
    # Other analysts have no tools (pure reasoning)
    assert not trading_analyst_agent.tools
    assert not execution_analyst_agent.tools
    assert not risk_analyst_agent.tools
