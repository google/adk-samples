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

"""set_workspace_window — let the agent anchor the session to project dir(s).

A lens over the shared workspace, not a security boundary: it changes which
slice relative paths and repo_overview/search_files default to. The agent
should set it once the conversation's project is clear, and ANNOUNCE the focus
so the user isn't surprised; the user can clear it with /workspace /.
"""

from __future__ import annotations

from typing import Any

from horizon.environment_context import active_environment
from horizon.tools._paths import path_under_root
from horizon.workspace_window import WORKSPACE_WINDOW_STATE_KEY


async def set_workspace_window(
    dirs: list[str], tool_context: Any | None = None
) -> dict[str, Any]:
    """Anchor THIS conversation to one or more workspace-relative project dirs.

    Once set, your relative file paths and `repo_overview`/`search_files`
    default to these dirs instead of the whole shared workspace — this is how
    you avoid conflating this project with the user's other projects. Pass an
    empty list to clear the focus (span the whole workspace again).

    You should set this when the user makes the project clear (e.g. "let's work
    on the daily-news-bot") and tell the user you've focused on it. You can
    still reach any other path by prefixing it with `/` (e.g.
    `read_file("/other-project/x.md")`).

    Args:
        dirs: Workspace-relative directory names (e.g. ["daily-news-bot"]).
            Empty list clears the focus.
    """
    state = getattr(tool_context, "state", None)
    if state is None:
        return {"success": False, "error": "no session state available"}

    cleaned = [
        d.strip().strip("/") for d in dirs if isinstance(d, str) and d.strip()
    ]
    if not cleaned:
        state[WORKSPACE_WINDOW_STATE_KEY] = []
        return {"success": True, "window": [], "message": "focus cleared"}

    root = active_environment().working_dir.resolve()
    for d in cleaned:
        resolved, err = path_under_root(d, root)
        if err is not None or resolved is None or not resolved.is_dir():
            return {"success": False, "error": f"directory not found: {d}"}

    state[WORKSPACE_WINDOW_STATE_KEY] = cleaned
    return {"success": True, "window": cleaned}
