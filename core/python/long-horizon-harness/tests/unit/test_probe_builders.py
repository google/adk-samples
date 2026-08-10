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

"""Unit tests for the agent-identity probe builders (no GCP)."""

from __future__ import annotations

from scripts.probes.probe_agent_identity import (
    agent_identity_deploy_kwargs,
    in_sandbox_probe_commands,
)


def test_deploy_kwargs_request_agent_identity():
    assert agent_identity_deploy_kwargs()["identity_type"] == "AGENT_IDENTITY"


def test_in_sandbox_probe_includes_metadata_email_and_gcp_call():
    cmds = in_sandbox_probe_commands()
    joined = "\n".join(cmds)
    assert "service-accounts/default/email" in joined  # who am I
    assert "gcloud projects list" in joined  # can I call GCP
