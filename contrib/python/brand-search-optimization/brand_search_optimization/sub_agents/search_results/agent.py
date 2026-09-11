# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines visual Search Results Agent powered by Gemini Computer Use."""

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.llm_request import LlmRequest

from ...shared_libraries import constants
from ...tools.browser_computer import get_computer_use_toolset
from . import prompt

# ADK's BaseComputer exposes the *legacy* Gemini 2.5 Computer Use function
# names (click_at, type_text_at, hover_at, ...). Gemini 3.x models — including
# the gemini-3.5-flash this recipe defaults to — emit the modern streamlined
# action names instead (click, type, move, ...), so without this remapping the
# model's function calls do not resolve against tools_dict and the agent stalls.
#
# Modern ENVIRONMENT_BROWSER action set, per
# https://ai.google.dev/gemini-api/docs/computer-use:
#   click, double_click, drag_and_drop, go_back, go_forward, hotkey, key_down,
#   key_up, middle_click, mouse_down, mouse_up, move, navigate, press_key,
#   right_click, scroll, take_screenshot, triple_click, type, wait
#
# Each alias is registered *in addition to* the legacy name, so the same build
# keeps working against legacy gemini-2.5-computer-use-preview-10-2025.
_TOOL_NAME_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    ("click_at", ("click",)),
    ("type_text_at", ("type",)),
    ("hover_at", ("move",)),
    ("scroll_document", ("scroll",)),
    ("scroll_at", ("scroll",)),
    ("current_state", ("take_screenshot",)),
    ("key_combination", ("press_key", "hotkey")),
]


async def adapt_computer_use_tools_callback(
    _callback_context: Any,
    llm_request: LlmRequest,
) -> None:
    """Registers modern Gemini 3.x action names for the Computer Use tools.

    Each alias points at the same tool object as the legacy name, so both
    resolve and the recipe works against legacy and modern models alike.
    """
    for method_name, aliases in _TOOL_NAME_ALIASES:
        tool = llm_request.tools_dict.get(method_name)
        if tool is None:
            continue
        for alias in aliases:
            llm_request.tools_dict.setdefault(alias, tool)


search_results_agent = LlmAgent(
    model=constants.MODEL,
    name="search_results_agent",
    description="Visually navigates retail search engines using Computer Use to observe top competitor product titles.",
    instruction=prompt.SEARCH_RESULT_AGENT_PROMPT,
    tools=[
        get_computer_use_toolset(),
    ],
    before_model_callback=adapt_computer_use_tools_callback,
    output_key="observed_search_results",
)
