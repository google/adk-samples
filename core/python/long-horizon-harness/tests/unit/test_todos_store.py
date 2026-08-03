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
from dataclasses import dataclass
from pathlib import Path

import pytest

from horizon.environment_context import set_active_environment


@dataclass
class _Exec:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class _FsEnv:
    """Minimal BaseEnvironment stub backed by a real temp directory.

    read_file / write_file hit the host FS under ``root``; execute runs the
    command with a real shell so the embedded directory-walker exercises the
    same code path as against LocalEnvironment.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def working_dir(self) -> Path:
        return self._root

    async def read_file(self, path: Path) -> bytes:
        return Path(path).read_bytes()

    async def write_file(self, path: Path, content) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")

    async def execute(self, command: str, *, timeout=None) -> _Exec:
        import subprocess

        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=False
        )
        return _Exec(
            exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
        )


@pytest.fixture
def env(tmp_path: Path) -> _FsEnv:
    e = _FsEnv(tmp_path)
    set_active_environment(e)
    return e


def _run(coro):
    return asyncio.run(coro)


# -----------------------------------------------------------------------------
# read_todos
# -----------------------------------------------------------------------------


def test_read_empty_folder_returns_empty_list(env):
    from horizon.tools.todos import read_todos

    assert _run(read_todos()) == []


def test_read_lists_in_filename_order(env):
    from horizon.tools.todos import read_todos, reconcile_todos

    _run(
        reconcile_todos(
            [
                {"id": "first", "content": "first", "status": "in_progress"},
                {"id": "second", "content": "second", "status": "pending"},
                {"id": "third", "content": "third", "status": "pending"},
            ]
        )
    )
    todos = _run(read_todos())
    assert [t["content"] for t in todos] == ["first", "second", "third"]
    assert todos[0]["status"] == "in_progress"


# -----------------------------------------------------------------------------
# reconcile_todos: files match the submitted list exactly
# -----------------------------------------------------------------------------


def test_reconcile_writes_one_file_per_todo(env, tmp_path):
    from horizon.tools.todos import reconcile_todos

    _run(
        reconcile_todos(
            [
                {"id": "alpha", "content": "alpha", "status": "pending"},
                {"id": "beta", "content": "beta", "status": "in_progress"},
            ]
        )
    )
    folder = tmp_path / "lha" / "todos"
    names = sorted(p.name for p in folder.glob("*.md"))
    assert names == ["001-alpha.md", "002-beta.md"]


def test_reconcile_deletes_orphans(env, tmp_path):
    from horizon.tools.todos import reconcile_todos

    _run(
        reconcile_todos(
            [
                {"id": "a", "content": "a", "status": "pending"},
                {"id": "b", "content": "b", "status": "pending"},
                {"id": "c", "content": "c", "status": "pending"},
            ]
        )
    )
    _run(reconcile_todos([{"id": "a", "content": "a", "status": "completed"}]))
    folder = tmp_path / "lha" / "todos"
    names = sorted(p.name for p in folder.glob("*.md"))
    assert names == ["001-a.md"]


def test_reconcile_empty_clears_folder(env, tmp_path):
    from horizon.tools.todos import reconcile_todos

    _run(reconcile_todos([{"id": "a", "content": "a", "status": "pending"}]))
    _run(reconcile_todos([]))
    folder = tmp_path / "lha" / "todos"
    assert sorted(folder.glob("*.md")) == []


def test_reconcile_is_idempotent(env, tmp_path):
    from horizon.tools.todos import reconcile_todos

    todos = [
        {"id": "a", "content": "a", "status": "pending", "priority": "high"},
        {"id": "b", "content": "b", "status": "in_progress"},
    ]
    _run(reconcile_todos(todos))
    folder = tmp_path / "lha" / "todos"
    snapshot = {p.name: p.read_text() for p in folder.glob("*.md")}
    _run(reconcile_todos(todos))
    again = {p.name: p.read_text() for p in folder.glob("*.md")}
    assert snapshot == again


def test_reconcile_ignores_foreign_files(env, tmp_path):
    from horizon.tools.todos import reconcile_todos

    folder = tmp_path / "lha" / "todos"
    folder.mkdir(parents=True)
    (folder / "README.txt").write_text("not a todo")
    _run(reconcile_todos([{"id": "a", "content": "a", "status": "pending"}]))
    assert (folder / "README.txt").exists()


# -----------------------------------------------------------------------------
# Session scoping
# -----------------------------------------------------------------------------


def test_session_scoped_read_isolates_sessions(env):
    from horizon.tools.todos import read_todos, reconcile_todos

    _run(
        reconcile_todos(
            [{"id": "a", "content": "a", "status": "pending"}], session_id="s1"
        )
    )
    assert [t["content"] for t in _run(read_todos(session_id="s1"))] == ["a"]
    assert _run(read_todos(session_id="s2")) == []


def test_session_scoped_files_live_under_session_folder(env, tmp_path):
    from horizon.tools.todos import reconcile_todos

    _run(
        reconcile_todos(
            [{"id": "a", "content": "a", "status": "pending"}], session_id="s1"
        )
    )
    folder = tmp_path / "lha" / "todos" / "s1"
    assert sorted(p.name for p in folder.glob("*.md")) == ["001-a.md"]


def test_unscoped_read_still_uses_flat_board(env, tmp_path):
    from horizon.tools.todos import read_todos, reconcile_todos

    _run(reconcile_todos([{"id": "a", "content": "a", "status": "pending"}]))
    assert [t["content"] for t in _run(read_todos())] == ["a"]
    assert (tmp_path / "lha" / "todos" / "001-a.md").exists()
