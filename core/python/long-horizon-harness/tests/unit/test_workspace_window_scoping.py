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
    """Minimal tool_context exposing .state, mirroring how terminal reads it."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "projA").mkdir()
    (tmp_path / "projB").mkdir()
    (tmp_path / "projA" / "a.py").write_text("# ALPHA marker\n")
    (tmp_path / "projB" / "b.py").write_text("# ALPHA marker\n")
    set_active_environment(LocalEnvironment(working_dir=tmp_path))
    return tmp_path


def _run(coro):
    return asyncio.run(coro)


def test_repo_overview_windowed_excludes_other_project(workspace):
    from horizon.tools.repo_overview import repo_overview

    ctx = _Ctx({WORKSPACE_WINDOW_STATE_KEY: ["projA"]})
    result = _run(repo_overview(tool_context=ctx))
    assert result["success"]
    overview = result["overview"]
    assert "a.py" in overview
    assert "b.py" not in overview


def test_repo_overview_scope_workspace_sees_both(workspace):
    from horizon.tools.repo_overview import repo_overview

    ctx = _Ctx({WORKSPACE_WINDOW_STATE_KEY: ["projA"]})
    result = _run(repo_overview(scope="workspace", tool_context=ctx))
    assert result["success"]
    assert "projA" in result["overview"]
    assert "projB" in result["overview"]


def test_repo_overview_no_window_unchanged(workspace):
    from horizon.tools.repo_overview import repo_overview

    result = _run(repo_overview(tool_context=_Ctx({})))
    assert result["success"]
    assert "projA" in result["overview"] and "projB" in result["overview"]


def test_search_files_windowed_only_matches_window(workspace):
    from horizon.tools.file_ops import search_files

    ctx = _Ctx({WORKSPACE_WINDOW_STATE_KEY: ["projA"]})
    result = _run(search_files("ALPHA", tool_context=ctx))
    assert result["success"]
    paths = [m["path"] for m in result["matches"]]
    assert any("projA" in p for p in paths)
    assert not any("projB" in p for p in paths)


def test_search_files_scope_workspace_matches_all(workspace):
    from horizon.tools.file_ops import search_files

    ctx = _Ctx({WORKSPACE_WINDOW_STATE_KEY: ["projA"]})
    result = _run(search_files("ALPHA", scope="workspace", tool_context=ctx))
    assert result["success"]
    paths = [m["path"] for m in result["matches"]]
    assert any("projA" in p for p in paths) and any("projB" in p for p in paths)


def test_write_then_read_relative_lands_in_window(workspace):
    from horizon.tools.file_ops import read_file, write_file

    ctx = _Ctx({WORKSPACE_WINDOW_STATE_KEY: ["projA"]})
    res = _run(write_file("note.md", "hello", tool_context=ctx))
    assert res["success"]
    assert (workspace / "projA" / "note.md").read_text() == "hello"
    back = _run(read_file("note.md", tool_context=ctx))
    assert back["success"] and "hello" in back["content"]


def test_read_slash_path_escapes_window(workspace):
    from horizon.tools.file_ops import read_file

    (workspace / "projB" / "only-b.txt").write_text("from B")
    ctx = _Ctx({WORKSPACE_WINDOW_STATE_KEY: ["projA"]})
    res = _run(read_file("/projB/only-b.txt", tool_context=ctx))
    assert res["success"] and "from B" in res["content"]


def test_no_window_relative_unchanged(workspace):
    from horizon.tools.file_ops import read_file, write_file

    res = _run(write_file("top.md", "x", tool_context=_Ctx({})))
    assert res["success"]
    assert (workspace / "top.md").read_text() == "x"
    assert _run(read_file("top.md", tool_context=_Ctx({})))["success"]


def test_view_file_relative_lands_in_window(workspace):
    from horizon.tools.view_file import ViewFileTool

    (workspace / "projA" / "doc.txt").write_text("alpha-doc")
    (workspace / "projB" / "doc.txt").write_text("beta-doc")

    class _ToolCtx:
        def __init__(self, state):
            self.state = state
            self.actions = type("A", (), {"artifact_delta": {}})()

        async def save_artifact(self, *, filename, artifact):
            return 0

    ctx = _ToolCtx({WORKSPACE_WINDOW_STATE_KEY: ["projA"]})
    res = _run(
        ViewFileTool().run_async(args={"path": "doc.txt"}, tool_context=ctx)
    )
    assert res["success"]
    # Resolved to projA/doc.txt, not projB/doc.txt.
    assert res["size_bytes"] == len("alpha-doc")


def test_first_write_into_subdir_seeds_window(workspace):
    from horizon.tools.file_ops import write_file

    ctx = _Ctx({})
    _run(write_file("projA/out.md", "x", tool_context=ctx))
    assert ctx.state[WORKSPACE_WINDOW_STATE_KEY] == ["projA"]


def test_root_level_write_does_not_seed(workspace):
    from horizon.tools.file_ops import write_file

    ctx = _Ctx({})
    _run(write_file("top.md", "x", tool_context=ctx))
    assert ctx.state.get(WORKSPACE_WINDOW_STATE_KEY) in (None, [])


def test_existing_window_not_overridden_by_write(workspace):
    from horizon.tools.file_ops import write_file

    ctx = _Ctx({WORKSPACE_WINDOW_STATE_KEY: ["projA"]})
    # With window projA, "projB/out.md" resolves UNDER projA (relative join),
    # so it lands at projA/projB/out.md — the window is not overridden.
    _run(write_file("projB/out.md", "x", tool_context=ctx))
    assert ctx.state[WORKSPACE_WINDOW_STATE_KEY] == ["projA"]
