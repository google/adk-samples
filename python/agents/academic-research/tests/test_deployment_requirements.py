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

"""Tests for Agent Engine deployment requirements."""

import pytest

from deployment import runtime_requirements


def test_deployment_requirements_pin_local_framework_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The remote runtime should use the locally installed SDK versions."""
    installed_versions = {
        "google-cloud-aiplatform": "1.143.0",
        "google-adk": "1.27.4",
    }
    monkeypatch.setattr(
        runtime_requirements,
        "version",
        installed_versions.__getitem__,
    )

    assert runtime_requirements.get_deployment_requirements() == [
        "google-cloud-aiplatform[agent_engines,adk]==1.143.0",
        "google-adk==1.27.4",
    ]
