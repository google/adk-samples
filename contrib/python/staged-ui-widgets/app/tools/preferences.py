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
"""Editing the shopper profile -- and the widget refresh that follows.

This tool is where the staging design earns its keep. Changing a price
ceiling should visibly re-rank the cards the shopper is looking at, but the
tool that built those cards is not being called this turn. Because the pick
payload and its original query are in session state, this tool can re-rank
and re-stage them, and the flush emits the updated carousel alongside the
confirmation. There is no way to express that by rendering inline from
whichever tool the model happened to call.

The write itself goes through ``profile.update_preference``, which coerces
and refuses. A preference store that accepts anything a model offers is a
ranking that silently stops working.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from ..profile import EDITABLE_FIELDS, PreferenceError, update_preference
from .picks import restage_picks


def update_shopper_preference(
    field: str,
    value: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Updates one stored shopper preference.

    Preferences persist across sessions. If a pick carousel is already on
    screen, it is re-ranked against the new preference in the same turn, so
    confirm the change and mention that the cards updated.

    Args:
      field: Which preference to set. One of: shoe_size, apparel_size,
        price_ceiling, favorite_brands, preferred_categories, colors,
        avoid_materials, needs_waterproof.
      value: The new value. Lists accept a comma-separated string;
        needs_waterproof accepts yes or no; price_ceiling accepts a number.

    Returns:
      What was stored, and how many pick cards were refreshed as a result.
    """
    try:
        label, stored = update_preference(tool_context.state, field, value)
    except PreferenceError as exc:
        return {
            "status": "rejected",
            "widget": None,
            "editable_fields": sorted(EDITABLE_FIELDS),
            "summary": str(exc),
        }

    refresh = restage_picks(tool_context)

    summary = f"Set {label} to {_readable(stored)}."
    if refresh.changed:
        summary += f" Re-ranked {refresh.cards} pick cards against it."
    elif refresh.cards:
        summary += (
            " The cards already on screen are unchanged by it -- say so "
            "rather than implying they updated."
        )

    return {
        "status": "ok",
        "widget": "picks" if refresh.changed else None,
        "field": field,
        "stored_value": stored,
        # Cards the shopper will see change, so ``0`` covers both "nothing was
        # on screen" and "the re-rank came out identical". The summary is what
        # distinguishes them, because that difference is a sentence, not a
        # number.
        "picks_refreshed": refresh.cards if refresh.changed else 0,
        "summary": summary,
    }


def _readable(value: Any) -> str:
    """Renders a stored value the way someone would say it back."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
