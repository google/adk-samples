# Copyright 2025 Google LLC
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

"""Proves the live tier genuinely sees the real environment (Q19 guardrail)."""

import os

import pytest

import app.config as config_module

pytestmark = pytest.mark.integration


def test_live_env_reaches_config():
    # GOOGLE_CLOUD_PROJECT is not an app.config attribute (verified against
    # app/config.py — the google-genai/Vertex SDK clients read it straight
    # from os.environ), so assert on the real environment instead.
    assert os.environ.get("GOOGLE_CLOUD_PROJECT") not in (
        None,
        "",
        "test-project",
    )
    assert config_module.GCS_BUCKET is None  # local-first pin (open item 2)


def test_db_is_an_isolated_copy(isolated_live_db):
    assert str(config_module.DB_PATH) == str(isolated_live_db)
    assert "test_campaigns_" in str(isolated_live_db)
