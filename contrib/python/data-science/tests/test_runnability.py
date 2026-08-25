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

from dotenv import load_dotenv

_RECIPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The recipe builds its whole agent graph at import time: agent.py calls
# load_dataset_config() and init_database_settings() at module scope, the
# analytics sub-agent constructs a VertexAiCodeExecutor, and every sub-agent
# reads its model from the environment. So the environment and the GCP client
# factories have to be in place *before* the import, not inside the test body --
# CI has no application default credentials and no dataset. These values also
# stand in for the placeholder .env CI writes from .env.example, whose
# YOUR_VALUE_HERE region Vertex AI rejects.
_ENV = {
    "GOOGLE_CLOUD_PROJECT": "test-project",
    "GOOGLE_CLOUD_LOCATION": "global",
    "BQ_COMPUTE_PROJECT_ID": "test-project",
    "BQ_DATA_PROJECT_ID": "test-project",
    "BQ_DATASET_ID": "test-dataset",
    "DATASET_CONFIG_FILE": os.path.join(
        _RECIPE_ROOT, "forecasting_sticker_sales_dataset_config.json"
    ),
}


def test_agent_runnability() -> None:
    """Verify agent.py imports and defines the expected globals."""
    for key, value in _ENV.items():
        os.environ.setdefault(key, value)
    # The model names come from .env.example so this test does not pin them;
    # override=False keeps the values set above.
    load_dotenv(os.path.join(_RECIPE_ROOT, ".env.example"), override=False)

    with (
        # chase_sql/llm_utils.py calls load_dotenv(override=True) at module
        # scope, which would replace everything set above with the
        # YOUR_VALUE_HERE placeholders from the .env CI writes. Disable dotenv
        # for the import so this test depends only on the values above --
        # AGENTS.md requires the unit tests to run without a .env present.
        patch("dotenv.load_dotenv"),
        patch(
            "google.auth.default", return_value=(MagicMock(), "test-project")
        ),
        # Stub only the extension lookup, not VertexAiCodeExecutor itself:
        # the agent's `code_executor` field is validated against
        # BaseCodeExecutor, so a bare mock in its place fails validation.
        patch(
            "google.adk.code_executors.vertex_ai_code_executor"
            "._get_code_interpreter_extension"
        ),
        patch("google.adk.tools.bigquery.client.get_bigquery_client"),
        # llm_utils.py also calls both of these at module scope. The patch
        # above means they now receive the values from _ENV rather than the
        # .env placeholders, so they would probably succeed on their own;
        # stubbing them keeps the import off Vertex AI entirely rather than
        # relying on init() staying side-effect-free without credentials.
        patch("google.cloud.aiplatform.init"),
        patch("vertexai.init"),
    ):
        import data_science.agent

    assert data_science.agent.root_agent is not None
