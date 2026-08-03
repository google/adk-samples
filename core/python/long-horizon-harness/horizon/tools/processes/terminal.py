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

"""Background-aware ``terminal`` tool — wraps the env-direct foreground
tool with two new parameters:

* ``background=True`` — spawn under the session's ``ProcessRegistry``
  and return ``{session_id, pid, status: 'running'}`` immediately.
* ``on_timeout='background' | 'kill'`` — when a foreground command exceeds
  ``timeout_s``, by default promote it to a background session (with
  partial output preserved) rather than killing.

Per-operation permission gating is handled centrally by ``permission_guard``
in the ``before_tool_callback`` chain, so this tool just runs the command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from horizon.environment.registry import resolve_registry
from horizon.environment_context import active_environment
from horizon.tools._output_overflow import make_preview
from horizon.tools._paths import path_under_root
from horizon.tools.terminal_exec import _emit_stream
from horizon.tools.terminal_exec import terminal as _foreground_terminal


async def _open_handle(command: str, cwd: Path) -> Any:
    """Spawn a background-process handle on the active environment.

    Returns a ``LocalProcessHandle`` on the local backend, a
    ``SandboxProcessHandle`` on the sandbox backend. Both satisfy the
    ``ProcessHandle`` surface used by ``process.py`` and the registry.
    """
    from horizon.secrets import secret_env

    env = active_environment()
    return await env.spawn_process(
        command, cwd=cwd, env=await secret_env() or None
    )


async def terminal(
    command: str,
    timeout_s: int = 30,
    cwd: str | None = None,
    background: bool = False,
    on_timeout: str = "background",
    tool_context: Any | None = None,
) -> dict[str, Any]:
    """Run a shell command — foreground (default) or backgrounded.

    Use for one-off shell ops: git status, ls, grep, running tests, dev
    servers. Output is capped per stream.

    Args:
        command: Shell command (passed to /bin/sh -c — POSIX sh, not bash/zsh).
        timeout_s: Seconds before a foreground command times out. Default 30.
        cwd: Working directory. Relative paths resolve under the agent's
            working directory; absolute paths must be inside it.
        background: If True, spawn under the session's process registry
            and return ``{session_id, pid, status: 'running'}`` immediately
            instead of blocking. Use for dev servers, long builds, test
            suites — anything you don't want blocking the agent turn.
            Then drive lifecycle via the ``process`` tool.
        on_timeout: ``'background'`` (default) auto-promotes a slow
            foreground command to a background session on timeout,
            preserving partial output. ``'kill'`` matches classic
            behavior: SIGKILL on timeout, no recovery.

    Foreground returns: ``{stdout, stderr, exit_code, timed_out, truncated}``.
    Background / promoted returns: ``{session_id, pid, status, ...}``
    (see ``process`` tool for inspection).
    """
    if not background and on_timeout == "kill":
        return await _foreground_terminal(
            command=command,
            timeout_s=timeout_s,
            cwd=cwd,
            tool_context=tool_context,
        )

    env = active_environment()
    root = env.working_dir.resolve()

    resolved_cwd = _resolve_cwd(cwd, root)
    if isinstance(resolved_cwd, dict):
        return resolved_cwd  # cwd-outside-root error envelope

    if background:
        return await _spawn_background(
            command=command,
            cwd=resolved_cwd,
            tool_context=tool_context,
        )

    return await _foreground_with_promote(
        command=command,
        timeout_s=timeout_s,
        resolved_cwd=resolved_cwd,
        tool_context=tool_context,
    )


def _resolve_cwd(cwd: str | None, root: Path) -> Path | dict[str, Any]:
    if cwd is None:
        return root
    resolved, err = path_under_root(cwd, root)
    if err is not None:
        return {
            "stdout": "",
            "stderr": err.replace("Path outside", "cwd outside"),
            "exit_code": -1,
            "timed_out": False,
            "truncated": False,
        }
    assert resolved is not None
    return resolved


async def _spawn_background(
    *, command: str, cwd: Path, tool_context: Any | None
) -> dict[str, Any]:
    handle = await _open_handle(command, cwd)
    resolve_registry(tool_context).register(handle)
    return {
        "session_id": handle.session_id,
        "pid": handle.pid,
        "status": "running",
        "command": command,
    }


async def _foreground_with_promote(
    *,
    command: str,
    timeout_s: int,
    resolved_cwd: Path,
    tool_context: Any | None,
) -> dict[str, Any]:
    """Foreground execution that promotes to a background session on timeout.

    Uses ``LocalProcessHandle`` (PTY-backed) from the start so partial
    output survives the foreground→background hand-off. Trade-off: PTY
    merges stdout and stderr into a single ``output`` field — the model
    loses stream separation for this path, but gains the ability to keep
    watching a long-running command. Use ``on_timeout='kill'`` to fall
    back to classic separate-stream semantics.
    """
    handle = await _open_handle(command, resolved_cwd)
    exit_code = await handle.wait(timeout=float(timeout_s))

    if exit_code is not None:
        data, _, _ = await handle.read()
        # No promotion needed; drop the handle without registering. Spill
        # oversized output to lha/tool-output/ (same recovery path as the
        # foreground tool) so the full text isn't lost behind the preview.
        output, truncated, overflow_path = await _emit_stream(
            data.decode("utf-8", errors="replace"), stream="stdout"
        )
        result: dict[str, Any] = {
            "stdout": output,
            "stderr": "",
            "exit_code": exit_code,
            "timed_out": False,
            "truncated": truncated,
        }
        if overflow_path:
            result["stdout_overflow_path"] = overflow_path
        return result

    # Still running past timeout — promote.
    data, _, _ = await handle.read()
    partial, _ = make_preview(data.decode("utf-8", errors="replace"))
    resolve_registry(tool_context).register(handle)
    return {
        "session_id": handle.session_id,
        "pid": handle.pid,
        "status": "running",
        "timed_out": True,
        "partial_output": partial,
        "command": command,
        "hint": (
            f"Foreground timeout exceeded; the command is now backgrounded "
            f"as session {handle.session_id}. Drive it via the process tool "
            "(poll/wait/kill/write)."
        ),
    }
