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
"""What widgets exist, and the state keys each one owns.

Declaring widgets up front buys three things a tool rendering inline cannot
have:

*A stable identity.* One ``name`` derives the widget id, the surface id, and
every state key. A tool cannot typo a key that it does not spell.

*A deterministic order.* ``WIDGET_SPECS`` is a tuple, and the flush walks it
in order, so a turn that stages picks and a comparison always emits them in
the same sequence -- regardless of the order the model happened to call the
tools in.

*One place to look.* When a widget does not appear, the answer is a gate in
``lifecycle.py`` applied to a key named here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Prefix for every key this recipe writes, so ``user:``-scoped profile data
# and framework keys are never confused with staging bookkeeping.
_NS = "ui"


@dataclass(frozen=True)
class StagedWidgetSpec:
    """One stageable widget.

    ``name`` is the only identifier a caller needs; everything else is
    derived. The three key scopes are deliberate and different:

    ``register_key`` is session-scoped, so a payload staged in one turn is
    still there several turns later when the shopper says "show me those
    again" -- the revival path never recomputes.

    ``dirty_key`` and ``suppress_key`` carry the ``temp:`` prefix, which ADK
    strips before persisting state. They are per-turn signals, and making
    them un-persistable means they cannot leak into the next turn and emit a
    widget nobody asked for.

    ``emitted_key`` is session-scoped and does double duty: it records that
    the current payload already went out, *and* -- because writing it is a
    state delta -- it is what forces ADK to produce an event at all. See
    ``lifecycle.emit_staged_widgets``.
    """

    name: str
    semantic_type: str

    @property
    def widget_id(self) -> str:
        """``UiWidget.id``. Unique per event, which the flush guarantees."""
        return f"{_NS}-{self.name}"

    @property
    def surface_id(self) -> str:
        """A2UI ``surfaceId``.

        Derived from the name rather than randomised, so re-showing a widget
        updates the surface a host already has instead of accumulating a new
        one per turn.
        """
        return f"{_NS}-surface-{self.name}"

    @property
    def register_key(self) -> str:
        """Session state holding the payload. Survives the turn."""
        return f"{_NS}:register:{self.name}"

    @property
    def dirty_key(self) -> str:
        """``temp:`` flag meaning "a tool staged this during this turn"."""
        return f"temp:{_NS}:dirty:{self.name}"

    @property
    def emitted_key(self) -> str:
        """Session flag meaning "the current payload has been sent"."""
        return f"{_NS}:emitted:{self.name}"

    @property
    def suppress_key(self) -> str:
        """``temp:`` veto for this turn, set by a tool or a callback."""
        return f"temp:{_NS}:suppress:{self.name}"


# Declaration order is emission order.
WIDGET_SPECS: tuple[StagedWidgetSpec, ...] = (
    StagedWidgetSpec(name="picks", semantic_type="product_picks"),
    StagedWidgetSpec(name="comparison", semantic_type="product_comparison"),
    StagedWidgetSpec(name="order", semantic_type="order_timeline"),
    StagedWidgetSpec(name="spend", semantic_type="spend_trend"),
)

_BY_NAME: dict[str, StagedWidgetSpec] = {s.name: s for s in WIDGET_SPECS}

# A shared widget id would make the flush raise, and a shared register would
# make two widgets overwrite each other. Both are typos, so catch them at
# import rather than mid-conversation.
if len(_BY_NAME) != len(WIDGET_SPECS):
    raise ValueError("WIDGET_SPECS contains duplicate names")


def spec_for(name: str) -> StagedWidgetSpec:
    """The spec for a widget name.

    Raises ``KeyError`` on an unknown name: a tool staging a widget that does
    not exist is a coding error, and failing loudly here is far cheaper than
    a widget that silently never appears.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        known = ", ".join(sorted(_BY_NAME))
        raise KeyError(
            f"unknown staged widget {name!r}; declared widgets are: {known}"
        ) from None


def all_specs() -> tuple[StagedWidgetSpec, ...]:
    """Every spec, in emission order."""
    return WIDGET_SPECS
