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

from pathlib import Path

import pytest

from horizon.environment.process import ProcessHandle
from horizon.environment_context import environment_provider, set_environment_provider


@pytest.fixture(autouse=True)
def _restore_provider():
    prev = environment_provider()
    yield
    set_environment_provider(prev)


@pytest.mark.asyncio
async def test_agent_platform_env_spawns_and_self_refreshes():
    from examples.agent_platform_backend import AgentPlatformEnvironment

    env = AgentPlatformEnvironment("u1")
    handle = await env.spawn_process("echo hi")
    assert isinstance(handle, ProcessHandle)
    assert await env.refresh_auth() is True
    env.mark_gone()
    assert await env.refresh_auth() is False


def test_example_needs_zero_core_edits():
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "examples/agent_platform_backend.py").read_text()
    assert "set_environment_provider" in src
    assert "from horizon.environment import Environment" in src
    for bad in ("horizon.conversation", "horizon.sandbox.provider", "session_start"):
        assert bad not in src
