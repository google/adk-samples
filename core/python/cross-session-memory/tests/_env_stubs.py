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

"""Shared stubs so the recipe's app modules can be imported in a test
environment with no `.env` file, no Application Default Credentials, and
no live GCP access.

Called from both `tests/conftest.py` (in-process tests) and
`tests/integration/_e2e_bootstrap.py` (subprocess uvicorn server).
"""

import os
from unittest.mock import MagicMock, patch


def install() -> None:
    """Set safe env-var defaults and patch GCP client factories.

    Safe to call multiple times; env vars use setdefault and patches are
    started once per process.
    """
    # Env vars every code path in the recipe needs at import time.
    os.environ.setdefault("MODEL_NAME", "gemini-3.5-flash")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
    # Signal to the recipe that it's running under an integration test so
    # code paths guarded by INTEGRATION_TEST skip live GCP calls.
    os.environ.setdefault("INTEGRATION_TEST", "TRUE")

    # google.auth.default(): return fake creds so vertexai.init and
    # google.cloud.aiplatform.initializer stop asking for ADC.
    import google.auth

    patch.object(
        google.auth,
        "default",
        return_value=(MagicMock(name="fake-credentials"), "test-project"),
    ).start()

    # vertexai.init(): no-op so agent_engine_app.set_up() doesn't call
    # out to the platform.
    import vertexai

    patch.object(vertexai, "init", MagicMock(name="vertexai.init")).start()

    # google.cloud.logging.Client(): return a stub whose .logger() gives
    # a MagicMock, so any log_struct calls are absorbed silently.
    from google.cloud import logging as google_cloud_logging

    patch.object(
        google_cloud_logging,
        "Client",
        MagicMock(name="google_cloud_logging.Client"),
    ).start()
