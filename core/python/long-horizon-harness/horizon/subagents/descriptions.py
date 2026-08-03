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

"""Per-turn rewrite of the delegate/agent tool descriptions.

ADK derives a ``FunctionDeclaration`` from each tool's docstring once, at
import — so the model never sees session-scoped facts (which skills are
loaded, which child archetypes exist). This ``before_model_callback``
rewrites the live declaration's ``description`` every turn, splicing in
the current skill catalog and the profile registry. The Python docstrings
stay static; only the model's runtime view changes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse

from horizon.subagents.delegate_builder import load_session_skill_catalog
from horizon.subagents.profiles import render_profiles_block

_SUBAGENT_TOOL_NAMES = frozenset({"delegate", "agent"})
_DYNAMIC_MARKER = "\n\n<!-- lha:dynamic -->\n\n"


def render_skills_block() -> str:
    catalog = load_session_skill_catalog()
    header = "## Skills you can pass to a child (skills=[...])"
    if not catalog:
        return f"{header}\n_No skills are currently loaded._"
    lines = []
    for name in sorted(catalog):
        skill = catalog[name]
        description = getattr(
            getattr(skill, "frontmatter", None), "description", ""
        )
        lines.append(f"- {name}: {description}")
    return "\n".join([header, *lines])


def _base_description(current: str | None) -> str:
    if not current:
        return ""
    return current.split(_DYNAMIC_MARKER, 1)[0]


def _build_suffix() -> str:
    return f"{render_skills_block()}\n\n{render_profiles_block()}"


def make_subagent_description_callback() -> Callable[
    [CallbackContext, LlmRequest], Awaitable[LlmResponse | None]
]:
    async def _callback(
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        config = getattr(llm_request, "config", None)
        tools = getattr(config, "tools", None) if config is not None else None
        if not tools:
            return None

        suffix = _build_suffix()
        for tool in tools:
            fds = getattr(tool, "function_declarations", None) or []
            for fd in fds:
                if fd.name not in _SUBAGENT_TOOL_NAMES:
                    continue
                base = _base_description(fd.description)
                fd.description = f"{base}{_DYNAMIC_MARKER}{suffix}"
        return None

    return _callback


subagent_description_callback = make_subagent_description_callback()


__all__ = [
    "make_subagent_description_callback",
    "render_skills_block",
    "subagent_description_callback",
]
