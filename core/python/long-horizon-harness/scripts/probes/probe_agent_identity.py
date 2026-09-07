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

"""Probe whether a sandbox can use an Agent-Identity (cert-bound) GCP token.

Builders are the source of truth for the deploy flag + in-sandbox checks; main()
exercises them live and is gated by RUN_SANDBOX_PROBE. Open question: bound
tokens are restricted to their "trusted runtime" and the docs don't cover
sandboxes.
"""

from __future__ import annotations

import os
import sys


def agent_identity_deploy_kwargs() -> dict[str, str]:
    """Kwargs to deploy the engine with Agent Identity.

    Maps to vertexai types.IdentityType.AGENT_IDENTITY at the agent-engine create
    step (scripts/provision_agent_engine.py); string form keeps this dependency-free.
    """
    return {"identity_type": "AGENT_IDENTITY"}


def in_sandbox_probe_commands() -> list[str]:
    """Shell commands to run *inside* the sandbox to answer the question."""
    meta = "http://metadata.google.internal/computeMetadata/v1/instance"
    return [
        # 1. Which identity does the sandbox see?
        f"curl -s -H 'Metadata-Flavor: Google' {meta}/service-accounts/default/email",
        # 2. Can the ambient/bound token actually call Google Cloud from here?
        "gcloud projects list --format='value(projectId)' --limit=1",
        # 3. Control: confirm no user creds / ADC file is present.
        "ls -la /workspace/lha/config/gcloud 2>&1 | head",
    ]


def main() -> int:
    if not os.environ.get("RUN_SANDBOX_PROBE"):
        print("set RUN_SANDBOX_PROBE=1 to run the live probe", file=sys.stderr)
        return 0
    print("Deploy engine with:", agent_identity_deploy_kwargs())
    print("Then run inside a sandbox:")
    for c in in_sandbox_probe_commands():
        print("  $", c)
    print(
        "\nInterpretation: if (2) succeeds the sandbox CAN use the identity token; "
        "if it 401s while (1) returns an email, the cert-bound token is rejected "
        "outside its trusted runtime — broker stays permanent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
