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
from unittest.mock import patch

from google.adk.agents import Agent
from google.adk.apps import App


def test_agent_runnability() -> None:
    """Verifies agent.py compiles and instantiates the agent successfully."""
    # Provide a dummy project so agent.py does not call google.auth.default(),
    # and mock vertexai.init to avoid a real GCP call — both happen at import
    # time, so they must be patched before the import.
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
    os.environ.setdefault("INTEGRATION_TEST", "TRUE")

    with patch("vertexai.init"):
        import app.agent

    assert app.agent.root_agent is not None
    assert isinstance(app.agent.root_agent, Agent)
    assert app.agent.root_agent.name == "root_agent"

    assert app.agent.app is not None
    assert isinstance(app.agent.app, App)
    assert app.agent.app.name == "app"
