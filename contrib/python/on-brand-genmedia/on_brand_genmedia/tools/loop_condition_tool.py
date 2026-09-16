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

import logging

from google.adk.tools import FunctionTool, ToolContext

from .. import config

logger = logging.getLogger(__name__)


def check_condition_and_escalate_tool(tool_context: ToolContext) -> dict:
    """Checks the loop termination condition and escalates if met or max count reached."""

    # Increment loop iteration count using state
    current_loop_count = tool_context.state.get("loop_iteration", 0)
    current_loop_count += 1
    tool_context.state["loop_iteration"] = current_loop_count

    # Define maximum iterations
    max_iterations = config.MAX_ITERATIONS

    # Get the condition result set by the sequential agent from state
    total_score = tool_context.state.get("total_score", 0)

    condition_met = total_score > config.SCORE_THRESHOLD

    response_message = f"Check iteration {current_loop_count}: Sequential condition met = {condition_met}. "

    # Check if the condition is met OR maximum iterations are reached
    if condition_met:
        logger.info(
            "Condition met. Setting escalate=True to stop the LoopAgent."
        )
        tool_context.actions.escalate = True
        response_message += "Condition met, stopping loop."
    elif current_loop_count >= max_iterations:
        logger.info(
            f"Max iterations ({max_iterations}) reached. Setting escalate=True to stop the LoopAgent."
        )
        tool_context.actions.escalate = True
        response_message += "Max iterations reached, stopping loop."
    else:
        logger.info(
            "Condition not met and max iterations not reached. Loop will continue."
        )
        response_message += "Loop continues."

    return {
        "status": "Evaluated scoring condition",
        "message": response_message,
    }


check_tool_condition = FunctionTool(func=check_condition_and_escalate_tool)
