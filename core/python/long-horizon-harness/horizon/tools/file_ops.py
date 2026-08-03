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

"""File operations: read, write, patch, search.

active ``BaseEnvironment`` (``LocalEnvironment`` on the host,
``SandboxEnvironment`` for Agent Runtime sandboxes). Multi-hunk patch
parsing, file-locking, and in-memory checkpoint snapshots are out of scope.
"""

from __future__ import annotations

import mimetypes
import os
import shlex
from pathlib import Path
from typing import Any

from horizon.environment_context import active_environment
from horizon.models.media import sniff_image_or_pdf_mime
from horizon.tools._paths import path_under_root
from horizon.tools._replacers import find_replacement
from horizon.workspace_window import (
    maybe_seed_window,
    resolve_in_window,
    window_dirs,
)

_MAX_READ_CHARS = 200_000
MAX_LINE_LENGTH = 2000
_LINE_TRUNC_SUFFIX = "... (line truncated)"
_MAX_DIAGNOSTICS = 20


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "error": message}


def _resolve_under_env(path: str) -> tuple[Path | None, str | None]:
    return path_under_root(path, active_environment().working_dir.resolve())


def _state_of(tool_context: Any | None) -> Any:
    return getattr(tool_context, "state", None)


def _resolve_in_session_window(
    path: str, tool_context: Any | None
) -> tuple[Path | None, str | None]:
    env_root = active_environment().working_dir.resolve()
    return resolve_in_window(
        path, env_root, window_dirs(_state_of(tool_context))
    )


def guess_mime(name: str) -> str:
    mime, _encoding = mimetypes.guess_type(name)
    return mime or "application/octet-stream"


def guess_mime_from_bytes(name: str, data: bytes | None) -> str:
    """Prefer the format the bytes actually are over what the extension claims.

    A file named ``.png`` that holds JPEG bytes (or a browser upload with an
    empty ``File.type``) is typed from its magic bytes so the model receives a
    correctly-typed blob and picks the right decoder. Falls back to the
    extension when the magic bytes aren't recognized.
    """
    return sniff_image_or_pdf_mime(data) or guess_mime(name)


_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".tiff",
        ".tif",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".wmv",
        ".flv",
        ".m4v",
        ".mpeg",
        ".mpg",
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".aac",
        ".m4a",
        ".wma",
        ".aiff",
        ".opus",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".rar",
        ".xz",
        ".z",
        ".tgz",
        ".iso",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".o",
        ".a",
        ".obj",
        ".lib",
        ".app",
        ".msi",
        ".deb",
        ".rpm",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".odt",
        ".ods",
        ".odp",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        ".pyc",
        ".pyo",
        ".class",
        ".jar",
        ".war",
        ".ear",
        ".node",
        ".wasm",
        ".rlib",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".mdb",
        ".idx",
        ".psd",
        ".ai",
        ".eps",
        ".sketch",
        ".fig",
        ".xd",
        ".blend",
        ".3ds",
        ".max",
        ".swf",
        ".fla",
        ".lockb",
        ".dat",
        ".data",
    }
)


def _has_binary_extension(path: str) -> bool:
    return Path(path).suffix.lower() in _BINARY_EXTENSIONS


def _denied_exact_paths(home: str) -> set[str]:
    return {
        os.path.realpath(p)
        for p in (
            os.path.join(home, ".ssh", "authorized_keys"),
            os.path.join(home, ".ssh", "id_rsa"),
            os.path.join(home, ".ssh", "id_ed25519"),
            os.path.join(home, ".ssh", "config"),
            os.path.join(home, ".bashrc"),
            os.path.join(home, ".zshrc"),
            os.path.join(home, ".profile"),
            os.path.join(home, ".bash_profile"),
            os.path.join(home, ".zprofile"),
            os.path.join(home, ".netrc"),
            os.path.join(home, ".pgpass"),
            os.path.join(home, ".npmrc"),
            os.path.join(home, ".pypirc"),
            "/etc/sudoers",
            "/etc/passwd",
            "/etc/shadow",
        )
    }


def _denied_prefixes(home: str) -> list[str]:
    return [
        os.path.realpath(p) + os.sep
        for p in (
            os.path.join(home, ".ssh"),
            os.path.join(home, ".aws"),
            os.path.join(home, ".gnupg"),
            os.path.join(home, ".kube"),
            "/etc/sudoers.d",
            "/etc/systemd",
            os.path.join(home, ".docker"),
            os.path.join(home, ".azure"),
            os.path.join(home, ".config", "gh"),
        )
    ]


def _is_write_denied(path: str) -> bool:
    home = os.path.realpath(os.path.expanduser("~"))
    resolved = os.path.realpath(os.path.expanduser(str(path)))
    if resolved in _denied_exact_paths(home):
        return True
    return any(resolved.startswith(prefix) for prefix in _denied_prefixes(home))


_SEARCH_DELIMITER = "###__LHA_SEARCH_RESULT__###"
_SEARCH_ERROR_KEY = "__lha_search_error__"


def _build_search_script(
    root: str, pattern: str, file_glob: str | None, limit: int
) -> str:
    return (
        "import fnmatch, json, os, re, sys\n"
        f"_root = {root!r}\n"
        f"_pattern = {pattern!r}\n"
        f"_file_glob = {file_glob!r}\n"
        f"_limit = {limit!r}\n"
        f"_error_key = {_SEARCH_ERROR_KEY!r}\n"
        f"_delim = {_SEARCH_DELIMITER!r}\n"
        f"_binary = {list(_BINARY_EXTENSIONS)!r}\n"
        "_binary = frozenset(_binary)\n"
        "try:\n"
        "    _re = re.compile(_pattern)\n"
        "except re.error as _exc:\n"
        "    print(_delim + json.dumps({_error_key: str(_exc)}) + _delim)\n"
        "    sys.exit(0)\n"
        "if not os.path.isdir(_root):\n"
        "    sys.stderr.write('search path not a directory: ' + _root + '\\n')\n"
        "    sys.exit(2)\n"
        "_matches = []\n"
        "for _dp, _dns, _fns in os.walk(_root):\n"
        "    _dns[:] = [d for d in _dns if not d.startswith('.')]\n"
        "    for _fn in _fns:\n"
        "        if _fn.startswith('.'):\n"
        "            continue\n"
        "        if _file_glob and not fnmatch.fnmatch(_fn, _file_glob):\n"
        "            continue\n"
        "        _dot = _fn.rfind('.')\n"
        "        if _dot != -1 and _fn[_dot:].lower() in _binary:\n"
        "            continue\n"
        "        _full = os.path.join(_dp, _fn)\n"
        "        try:\n"
        "            _mtime = os.stat(_full).st_mtime\n"
        "        except OSError:\n"
        "            continue\n"
        "        try:\n"
        "            with open(_full, 'r', encoding='utf-8', errors='replace') as _fh:\n"
        "                for _ln, _line in enumerate(_fh, start=1):\n"
        "                    _line = _line.rstrip('\\n')\n"
        "                    if _re.search(_line):\n"
        "                        _matches.append({'path': _full, 'line': _ln, 'text': _line, 'mtime': _mtime})\n"
        "                        if len(_matches) >= _limit:\n"
        "                            break\n"
        "        except OSError:\n"
        "            continue\n"
        "        if len(_matches) >= _limit:\n"
        "            break\n"
        "    if len(_matches) >= _limit:\n"
        "        break\n"
        "print(_delim + json.dumps(_matches) + _delim)\n"
    )


def extract_delimited(text: str, delimiter: str) -> str | None:
    """Pulls an embedded script's JSON payload out of its stdout, fenced by a
    sentinel the surrounding shell noise won't contain."""
    first = text.find(delimiter)
    if first == -1:
        return None
    after = first + len(delimiter)
    second = text.find(delimiter, after)
    if second == -1:
        return None
    return text[after:second]


def _format_numbered_lines(
    window: list[str], first_line_no: int, total_lines: int
) -> str:
    rendered: list[str] = []
    for i, raw_line in enumerate(window):
        line = raw_line
        if len(line) > MAX_LINE_LENGTH:
            line = raw_line[:MAX_LINE_LENGTH] + _LINE_TRUNC_SUFFIX
        rendered.append(f"{first_line_no + i}: {line}")

    body = "\n".join(rendered)
    last_line_no = first_line_no + len(window) - 1
    if last_line_no >= total_lines:
        trailer = f"End of file - total {total_lines} lines"
    else:
        trailer = (
            f"Showing lines {first_line_no}-{last_line_no} of {total_lines}. "
            f"Use offset={last_line_no + 1} to continue."
        )
    return f"{body}\n\n{trailer}"


async def read_file(
    path: str,
    *,
    offset: int = 1,
    limit: int = 500,
    tool_context: Any | None = None,
) -> dict[str, Any]:
    """Read a text file, optionally paginated by 1-indexed line offset.

    Relative paths resolve under your current workspace focus (see
    `/workspace`); prefix with `/` to reach the whole workspace.
    """
    import asyncio

    target, err = _resolve_in_session_window(path, tool_context)
    if err is not None:
        return _error(err)
    assert target is not None

    if _is_write_denied(str(target)):
        return _error(f"Read denied: {path} is on the protected paths list.")

    if _has_binary_extension(str(target)):
        return _error(f"Cannot read binary file: {path}")

    # Reject non-regular files (FIFOs, sockets, character devices) when the
    # path resolves on the host. ``Path.read_bytes`` on `/dev/urandom` would
    # block forever — better to fail fast. The check is skipped when the
    # path does not exist on the host (e.g. ``/workspace/...`` while
    # running against a SandboxEnvironment), where the env handles errors.
    if os.path.lexists(target) and not os.path.isfile(target):
        if os.path.isdir(target):
            return _error(f"Path is a directory, not a file: {path}")
        return _error(f"Not a regular file: {path}")

    try:
        raw = await active_environment().read_file(target)
    except FileNotFoundError:
        return _error(f"File not found: {path}")
    except IsADirectoryError:
        return _error(f"Path is a directory, not a file: {path}")
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except OSError as exc:
        return _error(f"Failed to read {path}: {exc}")

    text = raw.decode("utf-8", errors="replace")
    if text == "":
        return {"success": True, "content": ""}

    lines = text.splitlines()
    total_lines = len(lines)
    start = max(offset - 1, 0)
    if start >= total_lines:
        return {"success": True, "content": ""}
    end = start + max(limit, 0)
    window = lines[start:end]

    body = _format_numbered_lines(window, start + 1, total_lines)
    if len(body) > _MAX_READ_CHARS:
        return {
            "success": True,
            "content": body[:_MAX_READ_CHARS],
            "truncated": True,
        }
    return {"success": True, "content": body}


async def write_file(
    path: str, content: str, *, tool_context: Any | None = None
) -> dict[str, Any]:
    """Write content to path, creating parent directories as needed.

    Relative paths resolve under your current workspace focus (see
    `/workspace`); prefix with `/` to reach the whole workspace.
    """
    target, err = _resolve_in_session_window(path, tool_context)
    if err is not None:
        return _error(err)
    assert target is not None

    if _is_write_denied(str(target)):
        return _error(f"Write denied: {path} is on the protected paths list.")

    try:
        await active_environment().write_file(target, content)
    except OSError as exc:
        return _error(f"Failed to write {path}: {exc}")

    maybe_seed_window(
        _state_of(tool_context),
        target,
        active_environment().working_dir.resolve(),
    )

    result: dict[str, Any] = {"success": True, "path": str(target)}
    if str(target).endswith(".py"):
        diagnostics = await _post_edit_diagnostics(str(target))
        if diagnostics:
            result["diagnostics"] = diagnostics
    return result


async def _post_edit_diagnostics(path: str) -> str | None:
    """Best-effort ruff diagnostics for a just-written ``.py`` file.

    Returns a model-facing ``file:line: code message`` block (capped), or
    ``None`` when ruff is unavailable or produced nothing actionable. Never
    raises — a lint failure must not turn a successful edit into an error.
    """
    import asyncio
    import json

    command = (
        f"ruff check --output-format=json --force-exclude {shlex.quote(path)}"
    )
    try:
        result = await active_environment().execute(command, timeout=30.0)
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception:
        return None

    stdout = (result.stdout or "").strip()
    if not stdout:
        return None
    try:
        findings = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(findings, list) or not findings:
        return None

    lines: list[str] = []
    for item in findings[:_MAX_DIAGNOSTICS]:
        if not isinstance(item, dict):
            continue
        loc = item.get("location") or {}
        row = loc.get("row", "?")
        code = item.get("code") or ""
        message = item.get("message", "")
        prefix = f"{code} " if code else ""
        lines.append(f"{path}:{row}: {prefix}{message}".rstrip())

    if not lines:
        return None

    overflow = len(findings) - len(lines)
    body = "\n".join(lines)
    if overflow > 0:
        body += f"\n... and {overflow} more"
    return "ruff diagnostics (informational — the edit was applied):\n" + body


async def patch(
    path: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
    tool_context: Any | None = None,
) -> dict[str, Any]:
    """Replace old_string with new_string in path.

    By default old_string must occur exactly once; pass replace_all=True to
    swap every occurrence. The file is left untouched on any error. Relative
    paths resolve under your current workspace focus (see `/workspace`);
    prefix with `/` to reach the whole workspace.
    """
    target, err = _resolve_in_session_window(path, tool_context)
    if err is not None:
        return _error(err)
    assert target is not None

    if _is_write_denied(str(target)):
        return _error(f"Write denied: {path} is on the protected paths list.")

    env = active_environment()
    try:
        raw = await env.read_file(target)
    except FileNotFoundError:
        return _error(f"File not found: {path}")
    except IsADirectoryError:
        return _error(f"Path is a directory, not a file: {path}")
    except OSError as exc:
        return _error(f"Failed to read {path}: {exc}")

    original = raw.decode("utf-8", errors="replace")
    resolution = find_replacement(original, old_string, replace_all=replace_all)
    if resolution.error is not None:
        return _error(f"{resolution.error} (file: {path})")
    assert resolution.search is not None

    search = resolution.search
    updated = (
        original.replace(search, new_string)
        if replace_all
        else original.replace(search, new_string, 1)
    )

    try:
        await env.write_file(target, updated)
    except OSError as exc:
        return _error(f"Failed to write {path}: {exc}")

    result: dict[str, Any] = {
        "success": True,
        "path": str(target),
        "replacements": resolution.count,
    }
    if str(target).endswith(".py"):
        diagnostics = await _post_edit_diagnostics(str(target))
        if diagnostics:
            result["diagnostics"] = diagnostics
    return result


async def search_files(
    pattern: str,
    *,
    path: str = ".",
    file_glob: str | None = None,
    limit: int = 50,
    scope: str = "window",
    tool_context: Any | None = None,
) -> dict[str, Any]:
    """Search text files under your current workspace focus for `pattern`
    (Python regex). Pass `scope="workspace"` (or `path="/"`) to search the
    entire workspace across all projects.

    The first ``limit`` matches found (directory-walk order) are returned,
    ordered by source-file mtime (most recent first) — on a large tree the
    set is whichever matches the walk reached first, not a global top-N by
    mtime. Implemented as a single ``env.execute`` call that runs an embedded
    Python walker — same code path against ``LocalEnvironment`` and
    ``SandboxEnvironment``."""
    import json

    env_root = active_environment().working_dir.resolve()
    window = (
        [] if scope == "workspace" else window_dirs(_state_of(tool_context))
    )
    root, err = resolve_in_window(path, env_root, window)
    if err is not None:
        return _error(err)
    assert root is not None

    script = _build_search_script(str(root), pattern, file_glob, limit)
    command = f"python3 -c {shlex.quote(script)}"
    result = await active_environment().execute(command)
    if result.exit_code != 0:
        msg = result.stderr.strip() or "search failed"
        if "not a directory" in msg.lower() or "No such file" in msg:
            return _error(f"Search path invalid: {path}: {msg}")
        return _error(f"Search failed for {path}: {msg}")

    payload = extract_delimited(result.stdout, _SEARCH_DELIMITER)
    if payload is None:
        return _error(
            f"Search produced no result envelope for {path}; stderr={result.stderr!r}"
        )
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        return _error(f"Search produced invalid JSON: {exc}")

    if isinstance(parsed, dict) and _SEARCH_ERROR_KEY in parsed:
        return _error(f"Invalid regex pattern: {parsed[_SEARCH_ERROR_KEY]}")

    matches = sorted(parsed, key=lambda m: m.get("mtime", 0), reverse=True)
    return {"success": True, "matches": matches}
