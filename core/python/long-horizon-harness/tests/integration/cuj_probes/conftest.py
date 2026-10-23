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

"""Layer A integration probes — exercise the post-PR2 surface without GCP.

Every probe in this directory is in-process: FastAPI ``TestClient`` against
a router-only app, real ``InMemoryReminderStore`` / ``SubAgentRegistry``,
``verify_oauth2_token`` monkeypatched. No Vertex, no Cloud SQL, no
Cloud Scheduler — those land in Layer C.

Gated behind ``RUN_CUJ_PROBE=1`` so unit-test runs stay fast and so the
sandbox probes' ``RUN_SANDBOX_PROBE`` gate stays independent.
"""

from __future__ import annotations

import os

import pytest

_skipif_no_probe = pytest.mark.skipif(
    not os.environ.get("RUN_CUJ_PROBE"),
    reason="set RUN_CUJ_PROBE=1 to enable Layer A CUJ probes",
)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        if "tests/integration/cuj_probes" in str(item.fspath):
            item.add_marker(_skipif_no_probe)


pytestmark = pytest.mark.asyncio
