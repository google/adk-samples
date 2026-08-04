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

"""Tests for the extension's ``terminal`` wrapper.

Adds two parameters on top of the existing terminal tool:

* ``background: bool = False`` — spawn as a background process, return
  ``{session_id, pid, status: 'running'}`` without blocking.
* ``on_timeout: 'kill' | 'background' = 'background'`` — when a foreground
  command exceeds ``timeout_s``, by default we promote it to a background
  session (preserving partial output) instead of killing it.

Per-operation approval is handled centrally by ``permission_guard`` in the
``before_tool_callback`` chain, so the wrapper itself never gates in-body —
it just runs (foreground, background, or auto-promoted).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from horizon.environment import LocalEnvironment
from horizon.environment_context import (
    clear_active_environment,
    set_active_environment,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def env_root(tmp_path: Path) -> Iterator[Path]:
    working_dir = tmp_path / "ws"
    working_dir.mkdir()
    set_active_environment(LocalEnvironment(working_dir=working_dir))
    prev = os.getcwd()
    os.chdir(working_dir)
    try:
        yield working_dir
    finally:
        os.chdir(prev)
        clear_active_environment()


def _ctx(state: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        state=state if state is not None else {},
        tool_confirmation=None,
        actions=SimpleNamespace(skip_summarization=False),
        request_confirmation=lambda **_: None,
    )


class TestForegroundUnchanged:
    async def test_simple_echo_returns_stdout(self, env_root: Path) -> None:
        from horizon.tools.processes.terminal import terminal

        result = await terminal(command="echo hi", tool_context=_ctx())
        assert result["exit_code"] == 0
        assert "hi" in result["stdout"]
        assert "session_id" not in result  # plain foreground = no session

    async def test_default_path_spills_overflow_to_file(
        self, env_root: Path
    ) -> None:
        # The agent-facing default path (on_timeout='background') must spill
        # oversized completed output to lha/tool-output/ and return a pointer —
        # not silently drop it behind make_preview.
        from horizon.tools.processes.terminal import terminal

        result = await terminal(
            command=(
                'python3 -c "import sys; '
                "[print('filler-%06d' % i) for i in range(30000)]; "
                "print('TAIL_MARKER_xyz')\""
            ),
            tool_context=_ctx(),
        )
        assert result["truncated"] is True
        assert "session_id" not in result  # completed foreground, not promoted
        overflow_path = result.get("stdout_overflow_path")
        assert overflow_path, f"expected a spill pointer, got {result!r}"
        spilled = Path(overflow_path)
        assert spilled.parent == (env_root / "lha" / "tool-output")
        assert "TAIL_MARKER_xyz" in spilled.read_text()

    async def test_on_timeout_kill_matches_today(self, env_root: Path) -> None:
        from horizon.tools.processes.terminal import terminal

        result = await terminal(
            command="sleep 5",
            timeout_s=1,
            on_timeout="kill",
            tool_context=_ctx(),
        )
        assert result["timed_out"] is True
        assert result["exit_code"] != 0
        assert "session_id" not in result


class TestBackgroundSpawn:
    async def test_background_returns_session_id_immediately(
        self, env_root: Path
    ) -> None:
        from horizon.tools.processes.terminal import terminal

        ctx = _ctx()
        result = await terminal(
            command="sleep 5", background=True, tool_context=ctx
        )
        assert result["status"] == "running"
        assert result["session_id"].startswith("proc_")
        assert result["pid"] > 0

    async def test_background_handle_lands_in_registry(
        self, env_root: Path
    ) -> None:
        from horizon.tools.processes.process import process
        from horizon.tools.processes.terminal import terminal

        ctx = _ctx()
        spawn = await terminal(
            command="sleep 5", background=True, tool_context=ctx
        )
        listing = await process(action="list", tool_context=ctx)
        running_ids = [s["session_id"] for s in listing["running"]]
        assert spawn["session_id"] in running_ids

    async def test_background_does_not_block(self, env_root: Path) -> None:
        import time

        from horizon.tools.processes.terminal import terminal

        start = time.monotonic()
        await terminal(command="sleep 30", background=True, tool_context=_ctx())
        elapsed = time.monotonic() - start
        assert elapsed < 2.0


class TestAutoPromote:
    async def test_promote_on_timeout(self, env_root: Path) -> None:
        from horizon.tools.processes.terminal import terminal

        ctx = _ctx()
        result = await terminal(
            command="echo partial_marker; sleep 30",
            timeout_s=1,
            on_timeout="background",
            tool_context=ctx,
        )
        assert result["timed_out"] is True
        assert result["status"] == "running"
        assert result["session_id"].startswith("proc_")
        assert "partial_marker" in result["partial_output"]

    async def test_promoted_session_is_alive(self, env_root: Path) -> None:
        from horizon.tools.processes.process import process
        from horizon.tools.processes.terminal import terminal

        ctx = _ctx()
        result = await terminal(
            command="sleep 30",
            timeout_s=1,
            on_timeout="background",
            tool_context=ctx,
        )
        poll = await process(
            action="poll", session_id=result["session_id"], tool_context=ctx
        )
        assert poll["status"] == "running"

    async def test_default_on_timeout_is_background(
        self, env_root: Path
    ) -> None:
        from horizon.tools.processes.terminal import terminal

        result = await terminal(
            command="sleep 30",
            timeout_s=1,
            tool_context=_ctx(),
        )
        assert result["status"] == "running"
        assert "session_id" in result


class TestNoInBodyGating:
    async def test_background_spawn_does_not_self_gate(
        self, env_root: Path
    ) -> None:
        """The wrapper no longer gates in-body — a confirm-tier policy does
        not short-circuit background spawn. Per-operation approval is the
        central permission_guard's job (before the tool ever runs)."""
        from horizon.tools.processes.terminal import terminal

        overlay = env_root / ".lha" / "policies.jsonl"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.write_text(
            '{"canonical_tool_name": "terminal", "requires_confirmation": '
            '{"command": ["dangerous"]}}\n',
            encoding="utf-8",
        )

        ctx = _ctx()
        ctx.tool_confirmation = None
        result = await terminal(
            command="dangerous sleep 5", background=True, tool_context=ctx
        )
        assert "confirmation_required" not in result
        assert result["status"] == "running"
        assert result["session_id"].startswith("proc_")


class TestRegistrySharing:
    async def test_subsequent_calls_share_registry(
        self, env_root: Path
    ) -> None:
        from horizon.tools.processes.process import process
        from horizon.tools.processes.terminal import terminal

        ctx = _ctx()
        s1 = await terminal(
            command="sleep 5", background=True, tool_context=ctx
        )
        s2 = await terminal(
            command="sleep 5", background=True, tool_context=ctx
        )
        listing = await process(action="list", tool_context=ctx)
        ids = {s["session_id"] for s in listing["running"]}
        assert s1["session_id"] in ids
        assert s2["session_id"] in ids

    async def test_cleanup_kills_running_processes(
        self, env_root: Path
    ) -> None:
        """After the wrapper spawns a sleep, kill it via process(kill) and
        confirm it is reaped — sanity for the test fixture's own teardown."""
        from horizon.tools.processes.process import process
        from horizon.tools.processes.terminal import terminal

        ctx = _ctx()
        spawn = await terminal(
            command="sleep 30", background=True, tool_context=ctx
        )
        await asyncio.sleep(0.1)
        result = await process(
            action="kill", session_id=spawn["session_id"], tool_context=ctx
        )
        assert result["status"] == "killed"
