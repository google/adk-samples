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

"""Terminal tool: run a shell command, capture stdout/stderr, enforce timeout.

Kept to its core local-subprocess shape: sudo password caching, an env-sync
watchdog, PTY fallback, background-process rewriting, and exit-code
interpretation are all out of scope.

``terminal()`` delegates to the active ``BaseEnvironment.execute`` so a
``LocalEnvironment`` runs the shell on the host while a
``SandboxEnvironment`` routes through the Agent Runtime sandbox REPL —
same wire format either way.
"""

from __future__ import annotations

import logging
import shlex
from typing import Any

from horizon.environment_context import active_environment
from horizon.tools._output_overflow import make_preview, overflow_to_file
from horizon.tools._paths import path_under_root

logger = logging.getLogger(__name__)


async def _execute_with_secrets(
    env: Any, command: str, timeout: float, secrets: dict[str, str]
) -> Any:
    # Only the sandbox backend's execute() accepts an env kwarg; ADK's
    # LocalEnvironment does not, so pass secrets through only when supported.
    import inspect

    if secrets and "env" in inspect.signature(env.execute).parameters:
        return await env.execute(command, timeout=timeout, env=secrets)
    return await env.execute(command, timeout=timeout)


async def _emit_stream(
    text: str, *, stream: str
) -> tuple[str, bool, str | None]:
    """Return ``(payload, truncated, overflow_path)`` for one output stream.

    Small output is returned verbatim. Oversized output is spilled to a
    snapshot-persistent file and the preview+pointer is returned inline.
    """
    _, truncated = make_preview(text)
    if not truncated:
        return text, False, None
    result = await overflow_to_file(text, stream=stream)
    return f"{result.preview}\n\n{result.pointer}", True, result.path


async def terminal(
    command: str,
    timeout_s: int = 30,
    cwd: str | None = None,
    tool_context: Any | None = None,
) -> dict[str, Any]:
    """Run a shell command and return its output.

    Use this for one-off operations that don't have a dedicated tool: git
    status, ls, grep, running tests, etc. Output over ~50 KB or ~800 lines
    per stream is spilled to a file under lha/tool-output/ in your workspace
    and the result carries a preview plus a pointer telling you how to read
    the rest (view_file with offset/limit, grep, or a subagent). The command
    runs with a 30-second default timeout; raise timeout_s for legitimately
    slow commands.

    Args:
        command: Shell command to run (passed to /bin/sh -c). May contain
            pipes, redirects, and multi-line scripts.
        timeout_s: Seconds before the process is killed. Default 30.
        cwd: Working directory. Relative paths resolve under the agent's
            working directory; absolute paths must be inside it. Defaults
            to the agent's working directory.
    """
    env = active_environment()
    root = env.working_dir.resolve()

    if cwd is None:
        resolved_cwd = root
    else:
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
        resolved_cwd = resolved

    # BaseEnvironment.execute() runs at working_dir; emulate a per-call cwd
    # by prefixing `cd <path> && ...`. The shlex.quote keeps paths with
    # spaces / special chars safe.
    if resolved_cwd != root:
        full_command = f"cd {shlex.quote(str(resolved_cwd))} && {command}"
    else:
        full_command = command

    from horizon.secrets import secret_env

    secrets = await secret_env()
    result = await _execute_with_secrets(
        env, full_command, float(timeout_s), secrets
    )
    stdout, stdout_trunc, stdout_path = await _emit_stream(
        result.stdout, stream="stdout"
    )
    stderr, stderr_trunc, stderr_path = await _emit_stream(
        result.stderr, stream="stderr"
    )
    payload: dict[str, Any] = {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "truncated": stdout_trunc or stderr_trunc,
    }
    if stdout_path is not None:
        payload["stdout_overflow_path"] = stdout_path
    if stderr_path is not None:
        payload["stderr_overflow_path"] = stderr_path
    if result.timed_out:
        payload["hint"] = (
            f"Command exceeded the {timeout_s}s timeout and was killed. If it "
            f"is legitimately slow (not waiting on stdin), retry with a larger "
            f"timeout_s. For long-running work, use the process tool to run it "
            f"in the background instead."
        )
    return payload
