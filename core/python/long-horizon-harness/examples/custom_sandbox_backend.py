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

"""Plug a custom sandbox backend into Horizon via the Environment interface.

Run it:

    uv run uvicorn examples.custom_sandbox_backend:app --port 8001

Tools never touch the host directly — they go through the active ``Environment``,
resolved per session. ``Environment`` (``horizon.environment``) is Horizon's
runtime contract: a superset of ADK's ``BaseEnvironment`` that also declares
``list_directory`` / ``delete_file`` / ``make_dir`` / ``download_zip`` /
``upload_zip`` / ``spawn_process`` plus capability flags (``on_host_fs``).
``set_environment_provider(factory)`` installs a
``factory(user_id) -> Environment`` for the process; the built-in backends are
selected instead with ``LHA_ENVIRONMENT_BACKEND=local|sandbox``.

``StubEnvironment`` below is a clearly-marked, in-memory stub showing the surface
a real backend (e.g. GKE, Firecracker) implements. Constructs offline — no GCP
credentials. See ../docs/extending.md ("Sandbox backend").
"""

import os
from pathlib import Path

os.environ.setdefault("USE_IN_MEMORY_SESSION", "true")

from google.adk.environment import ExecutionResult

from horizon.environment import Environment
from horizon.environment_context import set_environment_provider


class StubEnvironment(Environment):
    """STUB: in-memory files, no real command execution. Replace the bodies with
    calls into your backend (container exec, remote filesystem)."""

    # Files live in the backend, not on the host: keep the host-fs short-circuits
    # off and route egress/overlay reads through the interface.
    on_host_fs = False

    def __init__(self, user_id: str) -> None:
        self._user_id = user_id
        self._files: dict[str, bytes] = {}

    @property
    def working_dir(self) -> Path:
        return Path("/workspace")

    async def execute(
        self, command: str, *, timeout: float | None = None
    ) -> ExecutionResult:
        # STUB: a real backend runs `command` in the user's isolated environment.
        return ExecutionResult(exit_code=0, stdout=f"[stub] would run: {command}")

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
        # STUB: a real backend lists `path`; return the shim shape
        # ({"name","kind","size","mtime"}, truncated).
        return [], False

    async def make_dir(self, path):
        return None  # STUB

    async def delete_file(self, path, *, recursive=False):
        self._files.pop(str(path), None)  # STUB

    async def download_zip(self, path):
        raise NotImplementedError("stub backend does not support zip export")

    async def upload_zip(self, path, data):
        raise NotImplementedError("stub backend does not support zip import")

    async def spawn_process(self, command, *, cwd=None, env=None):
        raise NotImplementedError("stub backend does not support background processes")


# Install the backend (one env per user_id) before the app builds. The provider
# is read when each session starts, so every user gets their own StubEnvironment.
set_environment_provider(lambda user_id: StubEnvironment(user_id))

from horizon.fast_api_app import app  # noqa: E402

__all__ = ["app"]
