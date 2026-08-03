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

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from horizon.environment import LocalEnvironment
from horizon.environment_context import set_active_environment
from horizon.workspace_window import WORKSPACE_WINDOW_STATE_KEY


class _Ctx:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


@pytest.fixture
def ws(tmp_path: Path):
    (tmp_path / "projA").mkdir()
    set_active_environment(LocalEnvironment(working_dir=tmp_path))
    return tmp_path


def _run(coro):
    return asyncio.run(coro)


def test_set_window_via_command(ws):
    from horizon.commands import _workspace

    ctx = _Ctx({})
    msg = _run(_workspace("projA", ctx))
    assert ctx.state[WORKSPACE_WINDOW_STATE_KEY] == ["projA"]
    assert "projA" in msg


def test_clear_window_via_command(ws):
    from horizon.commands import _workspace

    ctx = _Ctx({WORKSPACE_WINDOW_STATE_KEY: ["projA"]})
    _run(_workspace("/", ctx))
    assert ctx.state.get(WORKSPACE_WINDOW_STATE_KEY) in (None, [])


def test_nonexistent_dir_rejected(ws):
    from horizon.commands import _workspace

    ctx = _Ctx({})
    msg = _run(_workspace("nope", ctx))
    assert WORKSPACE_WINDOW_STATE_KEY not in ctx.state
    assert "nope" in msg.lower() or "not found" in msg.lower()


def test_no_arg_reports_current(ws):
    from horizon.commands import _workspace

    ctx = _Ctx({WORKSPACE_WINDOW_STATE_KEY: ["projA"]})
    msg = _run(_workspace("", ctx))
    assert "projA" in msg


def test_tool_sets_window(ws):
    from horizon.tools.workspace_window_tool import set_workspace_window

    ctx = _Ctx({})
    res = _run(set_workspace_window(["projA"], tool_context=ctx))
    assert res["success"]
    assert ctx.state[WORKSPACE_WINDOW_STATE_KEY] == ["projA"]


def test_tool_clears_window_on_empty(ws):
    from horizon.tools.workspace_window_tool import set_workspace_window

    ctx = _Ctx({WORKSPACE_WINDOW_STATE_KEY: ["projA"]})
    res = _run(set_workspace_window([], tool_context=ctx))
    assert res["success"]
    assert ctx.state[WORKSPACE_WINDOW_STATE_KEY] == []


def test_tool_rejects_missing_dir(ws):
    from horizon.tools.workspace_window_tool import set_workspace_window

    ctx = _Ctx({})
    res = _run(set_workspace_window(["nope"], tool_context=ctx))
    assert not res["success"]
    assert WORKSPACE_WINDOW_STATE_KEY not in ctx.state
