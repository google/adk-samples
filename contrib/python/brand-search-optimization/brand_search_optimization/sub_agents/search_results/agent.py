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

"""Defines Search Results Agent using ADK ComputerUseToolset.

Follows official ADK Computer Use pattern:
https://github.com/google/adk-python/tree/main/contributing/samples/multimodal/computer_use
"""

import functools
import os
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.computer_use.computer_use_toolset import (
    ComputerUseToolset,
)

from .playwright_computer import PlaywrightComputer
from .prompt import SEARCH_RESULT_AGENT_PROMPT


async def adapt_computer_use_tools_callback(
    _callback_context: Any,
    llm_request: LlmRequest,
) -> None:
    """Adapts BaseComputer tool names to standard Gemini Computer Use action names."""

    def make_adapter(new_name: str):
        def adapter(orig_func: Any):
            @functools.wraps(orig_func)
            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                return await orig_func(*args, **kwargs)

            wrapped.__name__ = new_name
            return wrapped

        return adapter

    adaptations = [
        ("click_at", "click"),
        ("type_text_at", "type"),
        ("hover_at", "hover"),
        ("scroll_document", "scroll"),
        ("current_state", "take_screenshot"),
        ("key_combination", "press_key"),
    ]

    for method_name, new_name in adaptations:
        if method_name in llm_request.tools_dict:
            orig_tool = llm_request.tools_dict[method_name]
            await ComputerUseToolset.adapt_computer_use_tool(
                method_name,
                make_adapter(new_name),
                llm_request,
            )
            # Retain original method name as an alias
            llm_request.tools_dict[method_name] = orig_tool


search_results_agent = Agent(
    model=os.getenv("MODEL_NAME"),
    name="search_results_agent",
    description=(
        "Inspects search engine result pages (SERPs) and audits brand "
        "ranking visibility using ADK Computer Use Toolset."
    ),
    instruction=SEARCH_RESULT_AGENT_PROMPT,
    tools=[ComputerUseToolset(computer=PlaywrightComputer())],
    before_model_callback=adapt_computer_use_tools_callback,
)
