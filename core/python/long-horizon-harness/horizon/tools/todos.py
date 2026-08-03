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

"""write_todos — file-based, compaction-surviving working memory.

Todos are files under ``<workspace>/lha/todos/<session_id>/`` (one markdown
file per todo, frontmatter + body), read and written through the
BaseEnvironment interface — on-brand with how lha treats skills (SKILL.md) and
memory (one file per fact). Riding the /workspace snapshot, each chat's board
survives event compaction and container teardown. Scoping by ``session_id``
keeps one chat's board out of another's prompt; other sessions' boards stay on
disk and remain readable on demand. ``session_id=None`` keeps the flat
``lha/todos/`` board (back-compat / all-sessions view).

Behavioral rules for the model live in ``write_todos``'s docstring; the
code only validates the status/priority enums and warns (never rejects)
when more than one item is in_progress.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from horizon.environment_context import active_environment
from horizon.tools._paths import path_under_root
from horizon.tools.file_ops import extract_delimited

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
VALID_STATUSES: frozenset[str] = frozenset(
    {STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_COMPLETED}
)
VALID_PRIORITIES: frozenset[str] = frozenset({"high", "medium", "low"})

TODOS_DIRNAME = "lha/todos"
_SLUG_MAX = 60
_LIST_DELIMITER = "###__LHA_TODOS__###"

_STATUS_MARKER: dict[str, str] = {
    STATUS_COMPLETED: "[x]",
    STATUS_IN_PROGRESS: "[~]",
    STATUS_PENDING: "[ ]",
}

_CAUTION = (
    "caution: more than one item is in_progress — keep exactly one "
    "in_progress at a time."
)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower()).strip("-")
    return (slug[:_SLUG_MAX].strip("-")) or "todo"


def filename_for(index: int, content: str) -> str:
    return f"{index + 1:03d}-{slugify(content)}.md"


def serialize_todo(todo: dict[str, Any]) -> str:
    lines = ["---", f"id: {todo['id']}", f"status: {todo['status']}"]
    if todo.get("priority"):
        lines.append(f"priority: {todo['priority']}")
    lines.append("---")
    lines.append(str(todo["content"]).strip())
    return "\n".join(lines) + "\n"


def parse_todo(text: str) -> dict[str, Any]:
    front: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    front[key.strip()] = value.strip()
            body = parts[2]

    todo: dict[str, Any] = {
        "content": body.strip(),
        "status": front.get("status", STATUS_PENDING),
    }
    if front.get("id"):
        todo["id"] = front["id"]
    if front.get("priority"):
        todo["priority"] = front["priority"]
    return todo


def _count_in_progress(todos: list[dict[str, Any]]) -> int:
    return sum(1 for t in todos if t.get("status") == STATUS_IN_PROGRESS)


def render_todos(todos: list[dict[str, Any]] | None) -> str:
    """Compact, byte-stable text view of the list for prompt injection.

    Returns "" for an empty/missing list so the caller omits the section.
    One line per todo: ``<marker> <content>`` with a trailing ``(high)``
    only for high-priority items.
    """
    if not todos:
        return ""

    lines = ["# Todos"]
    if _count_in_progress(todos) > 1:
        lines.append(_CAUTION)
    for todo in todos:
        marker = _STATUS_MARKER.get(todo.get("status", ""), "[ ]")
        content = str(todo.get("content", "")).strip()
        suffix = " (high)" if todo.get("priority") == "high" else ""
        lines.append(f"{marker} {content}{suffix}")
    return "\n".join(lines)


def session_id_of(context: Any | None) -> str | None:
    """Pull the ADK session id off a tool/callback context, or None."""
    inv = getattr(context, "_invocation_context", None)
    session = getattr(inv, "session", None)
    sid = getattr(session, "id", None)
    return sid or None


def _todos_dirname(session_id: str | None) -> str:
    return f"{TODOS_DIRNAME}/{session_id}" if session_id else TODOS_DIRNAME


def _todos_root(session_id: str | None = None) -> Path | None:
    root = active_environment().working_dir.resolve()
    resolved, err = path_under_root(_todos_dirname(session_id), root)
    if err is not None:
        return None
    return resolved


def _build_list_script(folder: str) -> str:
    return (
        "import json, os, re\n"
        f"_folder = {folder!r}\n"
        f"_delim = {_LIST_DELIMITER!r}\n"
        "_pat = re.compile(r'^\\d{3}-.*\\.md$')\n"
        "_out = []\n"
        "if os.path.isdir(_folder):\n"
        "    for _name in sorted(os.listdir(_folder)):\n"
        "        if not _pat.match(_name):\n"
        "            continue\n"
        "        try:\n"
        "            with open(os.path.join(_folder, _name), encoding='utf-8', "
        "errors='replace') as _fh:\n"
        "                _out.append({'name': _name, 'text': _fh.read()})\n"
        "        except OSError:\n"
        "            continue\n"
        "print(_delim + json.dumps(_out) + _delim)\n"
    )


async def _list_entries(session_id: str | None = None) -> list[dict[str, str]]:
    folder = _todos_root(session_id)
    if folder is None:
        return []
    script = _build_list_script(str(folder))
    result = await active_environment().execute(
        f"python3 -c {shlex.quote(script)}"
    )
    if result.exit_code != 0:
        return []
    payload = extract_delimited(result.stdout, _LIST_DELIMITER)
    if payload is None:
        return []
    try:
        entries = json.loads(payload)
    except json.JSONDecodeError:
        return []
    return [
        e
        for e in entries
        if isinstance(e, dict) and "name" in e and "text" in e
    ]


async def read_todos(session_id: str | None = None) -> list[dict[str, Any]]:
    """Read and parse the current todo files, in filename (list) order."""
    return [
        parse_todo(entry["text"]) for entry in await _list_entries(session_id)
    ]


async def reconcile_todos(
    todos: list[dict[str, Any]], session_id: str | None = None
) -> None:
    """Make the session's todo folder contain exactly ``todos`` (one file each).

    ``session_id`` selects ``lha/todos/<session_id>/``; ``None`` ⇒ the flat
    ``lha/todos/``. Each item must already carry an ``id`` and a normalized status.
    Existing ``NNN-*.md`` files not in the new list are removed first
    (orphan delete), then one file per item is written at its 1-based
    position prefix.
    """
    folder = _todos_root(session_id)
    if folder is None:
        return
    env = active_environment()

    keep = {filename_for(i, t["content"]) for i, t in enumerate(todos)}
    for entry in await _list_entries(session_id):
        if entry["name"] not in keep:
            await env.execute(
                f"rm -f {shlex.quote(str(folder / entry['name']))}"
            )

    for index, todo in enumerate(todos):
        await env.write_file(
            folder / filename_for(index, todo["content"]), serialize_todo(todo)
        )


def _validate_todo(item: Any, index: int) -> dict[str, Any] | str:
    """Return a normalized todo dict (with id), or an error string."""
    if not isinstance(item, dict):
        return f"todo #{index} must be an object with a 'content' field."

    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        return f"todo #{index} requires non-empty 'content'."
    content = content.strip()

    status = item.get("status") or STATUS_PENDING
    if status not in VALID_STATUSES:
        return (
            f"todo #{index} has invalid status {status!r}; "
            f"must be one of {sorted(VALID_STATUSES)}."
        )

    normalized: dict[str, Any] = {
        "id": slugify(content),
        "content": content,
        "status": status,
    }

    priority = item.get("priority")
    if priority is not None:
        if priority not in VALID_PRIORITIES:
            return (
                f"todo #{index} has invalid priority {priority!r}; "
                f"must be one of {sorted(VALID_PRIORITIES)}."
            )
        normalized["priority"] = priority

    return normalized


async def write_todos(
    todos: list[dict[str, Any]], tool_context: Any | None = None
) -> dict[str, Any]:
    """Replace your todo list with the full set of items provided.

    This is your externalized working memory. It is stored as files under
    ``lha/todos/<session_id>/`` in your workspace — one board per chat, so it
    persists across context compaction and survives container teardown without
    leaking into your other conversations. Use it to track multi-step work
    instead of relying on the transcript. Pass the ENTIRE list every call;
    it overwrites the stored list wholesale.

    Each item is an object:
    - content (str, required): brief, specific, actionable description.
    - status (str): one of "pending", "in_progress", "completed".
    - priority (str, optional): one of "high", "medium", "low".

    Rules:
    - Keep EXACTLY ONE item "in_progress" at a time. Mark the next item
      in_progress only after the current one is completed.
    - Mark an item "completed" ONLY after the work is actually done and
      verified (tests pass, command succeeded). Never mark completed on
      intent — "I will do X" is not "X is done".
    - If you get blocked or only partially finish an item, keep it
      in_progress and add a NEW pending item describing the blocker or the
      remaining work. Do not silently drop it.
    - Preserve any user-provided commands verbatim (flags, args, order) in
      the relevant item's content.
    - Use this for multi-step work (3+ steps). Skip it for trivial,
      single-step, or purely informational requests.

    Args:
        todos: The complete, updated todo list.
    """
    if not isinstance(todos, list):
        return {"success": False, "error": "todos must be a list of objects."}

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(todos):
        outcome = _validate_todo(item, index)
        if isinstance(outcome, str):
            return {"success": False, "error": outcome}
        normalized.append(outcome)

    await reconcile_todos(normalized, session_id_of(tool_context))

    result: dict[str, Any] = {
        "success": True,
        "todos": normalized,
        "rendered": render_todos(normalized),
    }
    if _count_in_progress(normalized) > 1:
        result["warning"] = (
            "More than one item is in_progress. Keep exactly one in_progress "
            "at a time; the list was still saved."
        )
    return result
