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

"""``terminal`` no longer self-gates — it always executes.

Per-operation approval moved to the central ``permission_guard``
(``before_tool_callback`` chain); its confirm/grant/decline round-trip is
covered by ``tests/unit/guardrails/test_permission_guard.py``. The terminal
tool itself just runs the command. These tests pin that passthrough: a
command runs regardless of any confirm-tier policy, never requests
confirmation in-body, and works with or without a ``tool_context``.

The environment is mocked in every test in this file, so even on a logic
bug NO real subprocess can run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from horizon.environment_context import (
    clear_active_environment,
    set_active_environment,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def hitl_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, AsyncMock]]:
    """Mock environment for terminal tests — no real subprocess can run."""
    from horizon.environment import LocalEnvironment

    working_dir = tmp_path / "ws"
    working_dir.mkdir()

    # Real env for policy hot-reload + path resolution.
    set_active_environment(LocalEnvironment(working_dir=working_dir))

    # Mock env exposed to terminal.py — execute is patched so even if the
    # tool logic accidentally falls through, nothing runs on the host.
    fake_execute = AsyncMock(
        return_value=SimpleNamespace(
            stdout="ok", stderr="", exit_code=0, timed_out=False
        )
    )
    mock_env = SimpleNamespace(working_dir=working_dir, execute=fake_execute)
    monkeypatch.setattr(
        "horizon.tools.terminal_exec.active_environment", lambda: mock_env
    )

    try:
        yield working_dir, fake_execute
    finally:
        clear_active_environment()


def _write_confirm_policy(working_dir: Path, pattern: str) -> None:
    overlay = working_dir / ".lha" / "policies.jsonl"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(
        '{"canonical_tool_name": "terminal", "requires_confirmation": '
        '{"command": ["' + pattern + '"]}}\n',
        encoding="utf-8",
    )


def _tool_context(
    *,
    tool_confirmation: SimpleNamespace | None = None,
    state: dict | None = None,
) -> SimpleNamespace:
    if state is None:
        state = {}
    return SimpleNamespace(
        state=state,
        tool_confirmation=tool_confirmation,
        actions=SimpleNamespace(skip_summarization=False),
        request_confirmation=MagicMock(),
    )


class TestTerminalDoesNotSelfGate:
    async def test_confirm_tier_command_executes_without_in_body_prompt(
        self, hitl_env: tuple[Path, AsyncMock]
    ):
        from horizon.tools.terminal_exec import terminal

        working_dir, fake_execute = hitl_env
        _write_confirm_policy(working_dir, "git push --force")
        ctx = _tool_context(tool_confirmation=None)

        result = await terminal(
            command="git push --force origin main", tool_context=ctx
        )

        # Gating is the central permission_guard's job now, not terminal's.
        ctx.request_confirmation.assert_not_called()
        fake_execute.assert_called_once()
        assert result["exit_code"] == 0

    async def test_safe_command_does_not_request_confirmation(
        self, hitl_env: tuple[Path, AsyncMock]
    ):
        from horizon.tools.terminal_exec import terminal

        working_dir, fake_execute = hitl_env
        _write_confirm_policy(working_dir, "git push --force")
        ctx = _tool_context(tool_confirmation=None)

        result = await terminal(command="echo hi", tool_context=ctx)

        ctx.request_confirmation.assert_not_called()
        fake_execute.assert_called_once()
        assert result["exit_code"] == 0

    async def test_no_tool_context_does_not_raise(
        self, hitl_env: tuple[Path, AsyncMock]
    ):
        # `terminal(..., tool_context=None)` is the call shape used by unit
        # helpers that don't care about gating. It must still work.
        from horizon.tools.terminal_exec import terminal

        working_dir, fake_execute = hitl_env
        _write_confirm_policy(working_dir, "git push --force")

        result = await terminal(command="echo hi", tool_context=None)
        assert result["exit_code"] == 0
        fake_execute.assert_called_once()
