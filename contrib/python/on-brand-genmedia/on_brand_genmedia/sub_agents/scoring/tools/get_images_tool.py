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

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


async def get_image(tool_context: ToolContext):
    artifact_name = (
        f"generated_image_{tool_context.state.get('loop_iteration', 0)}.png"
    )
    try:
        logger.debug("Entered the get_image function")
        logger.debug(f"artifact_name: {artifact_name}")
        image_part = await tool_context.load_artifact(artifact_name)
        logger.debug("artifact loaded successfully")

        return {
            "status": "success",
            "message": f"Image artifact {artifact_name} successfully loaded.",
            "image": image_part,
        }
    except Exception as e:
        logger.error(
            f"Error loading artifact {artifact_name}: {e!s}", exc_info=True
        )
        return {
            "status": "error",
            "message": f"Error loading artifact {artifact_name}: {e!s}",
        }
