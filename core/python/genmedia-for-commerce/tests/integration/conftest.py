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

"""Shared fixtures for integration tests.

Integration tests require real GCP credentials and a configured project.
Tests are automatically skipped when the environment is not set up
(e.g. in CI without service-account keys or a valid API key).
"""

import os

import pytest


def _real_credentials_configured() -> bool:
    """Return True when at least one non-placeholder credential is present.

    The CI workflow copies .env.example → .env, which contains placeholder
    values like ``<TODO: update-this-value>``.  Those are truthy strings but
    are not valid credentials.  We treat any value starting with ``<`` as a
    placeholder.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if project and not project.startswith("<"):
        return True
    if api_key and not api_key.startswith("<"):
        return True
    return False


_SKIP_REASON = (
    "Integration tests require real GCP credentials. "
    "Set GOOGLE_CLOUD_PROJECT to a valid project ID or GOOGLE_API_KEY to a "
    "valid Gemini API key before running."
)


@pytest.fixture(autouse=True)
def skip_without_real_credentials() -> None:
    """Auto-skip every integration test when credentials are not configured."""
    if not _real_credentials_configured():
        pytest.skip(_SKIP_REASON)
