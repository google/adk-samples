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
"""The agent: six tools that stage, two callbacks that bracket the turn.

``before_agent_callback`` makes sure a profile exists.
``after_agent_callback`` flushes whatever the tools staged. Nothing in
between renders anything.

### The trap this file exists to demonstrate

``after_agent_callback`` returning ``None`` normally means "I did not override
the reply", which is what a callback that only emits widgets wants to say. But
ADK creates an event for a callback only when it returned content **or** the
callback's state changed (``agents/base_agent.py:564-582``):

    if after_agent_callback_content:
        return Event(..., actions=callback_context._event_actions)
    if callback_context.state.has_delta():
        return Event(..., actions=callback_context._event_actions)
    return None

``render_ui_widget`` appends to ``_event_actions`` -- not to state. So a
callback that renders three widgets and writes nothing produces no event, no
widgets reach the client, and nothing raises. It looks exactly like a turn in
which no widget was staged.

``emit_staged_widgets`` writes an emitted flag per widget, which serves as the
dedupe record *and* the state delta that forces the event out.
``tests/unit/test_event_delivery.py`` locks the behaviour down with a
runner-level regression test, because the failure mode is silent and a future
refactor could reintroduce it without breaking anything visible.
"""

from __future__ import annotations

import logging
import os

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from .profile import PROFILE_KEY, load_profile, save_profile
from .prompt import build_instruction
from .staging import emit_staged_widgets, log_flush
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# Model default matches .env.example, so `adk run app` works from a fresh
# clone with only a credential set.
_DEFAULT_MODEL = "gemini-3.5-flash"


def seed_profile(callback_context: CallbackContext) -> None:
    """Ensures the shopper profile exists before the model runs.

    ``user:``-scoped state persists across sessions, so this writes only on a
    genuinely first visit. Writing unconditionally would overwrite
    preferences the shopper set last week -- and, because any state write
    forces an event, would also emit a pointless event on every single turn.
    """
    if PROFILE_KEY not in callback_context.state:
        save_profile(
            callback_context.state, load_profile(callback_context.state)
        )
        logger.info("seeded default shopper profile")


def flush_widgets(callback_context: CallbackContext) -> None:
    """Emits every staged widget that clears the gates.

    Returns ``None`` so the model's reply stands unmodified. The event that
    carries the widgets exists because ``emit_staged_widgets`` wrote state --
    see this module's docstring.
    """
    outcomes = emit_staged_widgets(callback_context)
    log_flush(outcomes)


def create_agent() -> Agent:
    """Creates a fresh, isolated instance of the Agent."""
    return Agent(
        name="root_agent",
        model=Gemini(
            model=os.getenv("MODEL_NAME", _DEFAULT_MODEL),
            retry_options=types.HttpRetryOptions(attempts=3),
        ),
        instruction=build_instruction,
        tools=ALL_TOOLS,
        before_agent_callback=seed_profile,
        after_agent_callback=flush_widgets,
    )


root_agent = create_agent()

app = App(
    root_agent=root_agent,
    name="app",
)
