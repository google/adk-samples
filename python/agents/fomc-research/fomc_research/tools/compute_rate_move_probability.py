# Copyright 2025 Google LLC
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

"""'compute_rate_move_probability' tool for FOMC Research sample agent."""

import logging

from google.adk.tools import ToolContext

from ..shared_libraries import price_utils

logger = logging.getLogger(__name__)


def compute_rate_move_probability_tool(
    tool_context: ToolContext,
) -> dict[str, str]:
    """Computes the probabilities of rate moves for the requested meeting date.

    Args:
      tool_context: ToolContext object.

    Returns:
      A dict with "status" and (optional) "error_message" keys.
    """
    meeting_date = tool_context.state["requested_meeting_date"]
    logger.debug("Computing rate move probabilities for %s", meeting_date)
    prob_result = price_utils.compute_probabilities(meeting_date)
    if prob_result["status"] != "OK":
        return {"status": "ERROR", "message": prob_result["message"]}
    probs = prob_result["output"]
    logger.debug("Rate move probabilities: %s", probs)
    tool_context.state.update({"rate_move_probabilities": probs})
    return {"status": "OK"}
