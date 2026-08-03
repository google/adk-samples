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

import pytest
from google.adk.environment import ExecutionResult

from horizon.environment import LocalEnvironment
from horizon.environment_context import set_active_environment
from horizon.secrets import set_secret_store
from horizon.secrets.store import SecretManagerStore
from tests.unit.test_secret_store import FakeSecretClient


class RecordingEnv(LocalEnvironment):
    def __init__(self, working_dir):
        super().__init__(working_dir=working_dir)
        self.last_env = None

    @property
    def owner(self):  # type: ignore[override]
        return "alice@x"

    async def execute(self, command, *, timeout=None, env=None):
        self.last_env = env
        return ExecutionResult(
            exit_code=0, stdout="", stderr="", timed_out=False
        )


@pytest.fixture()
def wired_store():
    store = SecretManagerStore(client=FakeSecretClient(), project_id="proj")
    set_secret_store(store)
    yield store
    set_secret_store(None)


@pytest.mark.asyncio
async def test_terminal_injects_secrets_into_execute(wired_store, tmp_path):
    await wired_store.set_secret("alice@x", "OPENAI_API_KEY", "sk-1")
    env = RecordingEnv(working_dir=tmp_path)
    set_active_environment(env)

    from horizon.tools.terminal_exec import terminal

    await terminal("echo hi", timeout_s=5)
    assert env.last_env == {"OPENAI_API_KEY": "sk-1"}
