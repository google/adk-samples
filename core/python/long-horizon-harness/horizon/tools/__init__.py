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

"""Core tools for the Horizon agent.

Docstring policy for every tool registered on the root agent (enforced by
tests/unit/test_prompt_budget.py): simple tool description <= 400 chars;
action-dispatch tool (process, subagent, artifact, routine) <= 900. Line 1
states what the tool does; document an arg only when its name and type
don't already carry the meaning. No "Use when" essays, no worked examples,
no comparisons to other tools — a cross-tool choice (this tool vs that one)
belongs in the system prompt's TOOL_ROUTING_GUIDANCE
(horizon/conversation/system_prompt.py), not in either tool's description.
"""

from horizon.tools.file_ops import edit, search_files, write

# `bash` / `process` (the registered tools) live in `horizon.tools.processes`;
# the foreground executor they wrap is `horizon.tools.terminal_exec` (kept
# under its pre-rename name there — it's an internal helper, not a
# registered tool, and dozens of tests import it directly by that name).
# Neither `bash` nor `process` is re-exported here — import from those
# modules directly.
#
# `read_file` is no longer a registered tool (merged into `ReadTool`,
# horizon/tools/read.py) but stays a module-level helper in `file_ops.py` for
# ReadTool's text branch and the existing file_ops test suite — import it from
# `horizon.tools.file_ops` directly, not from this package.

__all__ = [
    "edit",
    "search_files",
    "write",
]
