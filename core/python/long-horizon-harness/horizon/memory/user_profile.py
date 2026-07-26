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

"""USER profile backed by Memory Bank native Structured Profiles.

The dream-review pass generates a structured profile server-side (see
``horizon/memory/dream_review.py``); here we read it back via
``retrieve_profiles`` and render it for the system prompt.
``on_session_start_callback`` loads it once per session into
``state['user_profile']``, which ``render_user_profile`` formats into a
``## User Profile`` block in the volatile tier.
"""

from __future__ import annotations

import logging
from typing import Any

# engine_resource_name re-exported for callers/tests that referenced it here
# before the adapter boundary was introduced.
from horizon.memory.adapter import engine_resource_name, memory_adapter

logger = logging.getLogger(__name__)


def _render_profile(profile: dict[str, Any]) -> str:
    """Render a structured profile dict into a readable block."""
    lines: list[str] = []
    summary = str(profile.get("summary") or "").strip()
    if summary:
        lines.append(summary)
    fields: list[tuple[str, str]] = []
    if role := str(profile.get("role") or "").strip():
        fields.append(("Role", role))
    if interests := [
        str(i).strip() for i in profile.get("interests") or [] if str(i).strip()
    ]:
        fields.append(("Interests", ", ".join(interests)))
    if style := str(profile.get("working_style") or "").strip():
        fields.append(("Working style", style))
    facts = [
        str(f).strip() for f in profile.get("durable_facts") or [] if str(f).strip()
    ]
    if fields or facts:
        if lines:
            lines.append("")
        lines.extend(f"- **{label}:** {value}" for label, value in fields)
        lines.extend(f"- {fact}" for fact in facts)
    return "\n".join(lines).strip()


async def load_user_profile(
    *,
    memory_service: Any,
    app_name: str,
    user_id: str,
) -> str:
    """Return the user's structured profile as text, or '' if none/unavailable."""
    if not app_name or not user_id:
        return ""
    profile = await memory_adapter(memory_service).retrieve_profile(
        app_name=app_name, user_id=user_id
    )
    if not profile:
        return ""
    return _render_profile(profile)


def render_user_profile(state: dict | None) -> str:
    """Return a ``## User Profile`` block for state['user_profile'], else ''."""
    if not state:
        return ""
    profile = state.get("user_profile")
    if not profile or not str(profile).strip():
        return ""
    return f"## User Profile\n\n{str(profile).strip()}"


__all__ = [
    "engine_resource_name",
    "load_user_profile",
    "render_user_profile",
]
