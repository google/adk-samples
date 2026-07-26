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

"""Plug an agent-platform backend into Horizon via the Environment interface.

Unlike ``custom_sandbox_backend.py`` (files only), this shows the two pieces a
managed platform needs: a real ``spawn_process`` returning a ``ProcessHandle``
(imported from ``horizon.environment``), and per-turn ``refresh_auth`` for
short-lived platform tokens. No provider, no core edits.

Run it:

    uv run uvicorn examples.agent_platform_backend:app --port 8001
"""

import os
import time
from pathlib import Path

os.environ.setdefault("USE_IN_MEMORY_SESSION", "true")

from google.adk.environment import ExecutionResult

from horizon.environment import Environment
from horizon.environment.process import (
    ProcessHandle,
)
from horizon.environment_context import set_environment_provider


class _StubHandle:
    """STUB: a real backend returns a handle proxying its container process."""

    def __init__(self, command: str) -> None:
        self.session_id = f"proc-{id(self)}"
        self.pid = 4242
        self.command = command
        self.started_at = time.time()
        self.finished_at: float | None = self.started_at

    @property
    def is_running(self) -> bool:
        return False

    @property
    def exit_code(self) -> int | None:
        return 0

    @property
    def output_size(self) -> int:
        return 0

    @property
    def idle_seconds(self) -> float:
        return 0.0

    async def read(self, offset=0, limit=None):
        return b"", 0, 0

    async def write(self, data):
        return None

    async def kill(self):
        return None

    async def wait(self, timeout=None):
        return 0


class AgentPlatformEnvironment(Environment):
    """STUB: in-memory files + a stub process; replace the bodies with calls into
    your platform (container exec, remote fs, token mint)."""

    on_host_fs = False

    def __init__(self, user_id: str) -> None:
        self._user_id = user_id
        self._files: dict[str, bytes] = {}
        self.refresh_calls = 0
        self._gone = False

    @property
    def working_dir(self) -> Path:
        return Path("/workspace")

    async def execute(
        self, command: str, *, timeout: float | None = None
    ) -> ExecutionResult:
        return ExecutionResult(exit_code=0, stdout=f"[platform] {command}")

    async def read_file(self, path: Path) -> bytes:
        try:
            return self._files[str(path)]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def write_file(self, path: Path, content: str | bytes) -> None:
        self._files[str(path)] = (
            content.encode() if isinstance(content, str) else content
        )

    async def list_directory(self, path, *, limit):
        return [], False

    async def make_dir(self, path):
        return None

    async def delete_file(self, path, *, recursive=False):
        self._files.pop(str(path), None)

    async def download_zip(self, path):
        return b""

    async def upload_zip(self, path, data):
        return None

    async def spawn_process(self, command, *, cwd=None, env=None) -> ProcessHandle:
        return _StubHandle(command)

    async def refresh_auth(self) -> bool:
        # A managed platform re-mints a short-lived token here each turn.
        self.refresh_calls += 1
        return not self._gone

    def mark_gone(self) -> None:
        self._gone = True


def __getattr__(name: str) -> object:
    # Register the provider + build the served app only when ``app`` is actually
    # accessed (e.g. ``uvicorn examples.agent_platform_backend:app``), so merely
    # importing AgentPlatformEnvironment has no global side effects.
    if name == "app":
        set_environment_provider(lambda user_id: AgentPlatformEnvironment(user_id))
        from horizon.fast_api_app import app as _app

        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgentPlatformEnvironment", "app"]  # noqa: F822  (app is a lazy __getattr__ attribute)
