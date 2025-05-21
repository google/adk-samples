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

"""'store_state' tool for FOMC Research sample agent"""

import logging
import typing

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


def store_state_tool(
    state: dict[str, typing.Any], tool_context: ToolContext
) -> dict[str, str]:
    """Stores new state values in the ToolContext.

    Args:
      state: A dict of new state values.
      tool_context: ToolContext object.

    Returns:
      A dict with "status" and (optional) "error_message" keys.
    """
    logger.info("store_state_tool(): %s", state)
    tool_context.state.update(state)
    return {"status": "ok"}
