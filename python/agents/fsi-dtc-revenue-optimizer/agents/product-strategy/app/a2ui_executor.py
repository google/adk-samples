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

"""Custom A2A executor that converts A2UI JSON to A2A DataParts.

The ADK's built-in A2aAgentExecutor passes LLM text through as plain A2A
TextParts.  This custom executor runs the ADK runner, parses the
``---a2ui_JSON---`` delimiter from the final text response, and constructs
proper ``a2a.types.DataPart`` objects with MIME type
``application/json+a2ui`` so Gemini Enterprise renders them as A2UI
components.

Modeled after the working ``a2a_a2ui_sample`` reference implementation.
"""

import json
import logging
import uuid

from a2a import types as a2a_types
from a2a import utils as a2a_utils
from a2a.server import agent_execution
from a2a.server import events as a2a_events
from a2a.server import tasks as a2a_tasks
from a2a.utils import errors as a2a_errors
from google.adk import runners
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

_A2UI_DELIMITER = "---a2ui_JSON---"


class A2UIAgentExecutor(agent_execution.AgentExecutor):
    """Agent executor that creates A2A DataParts from A2UI JSON output."""

    def __init__(self, runner: runners.Runner):
        self._runner = runner
        self._user_id = "remote_agent"

    async def execute(
        self,
        context: agent_execution.RequestContext,
        event_queue: a2a_events.EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task
        logger.info("A2UI executor — query: %s", query)

        if not task:
            if not context.message:
                return
            task = a2a_utils.new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = a2a_tasks.TaskUpdater(event_queue, task.id, task.context_id)
        session_id = task.context_id

        # Detect video-only requests and isolate session to prevent
        # replaying previous strategic reports as context.
        # Only "video" (and "media") are anchors — broad verbs like
        # "generate" / "create" alone would cause false positives
        # (e.g. "create a pricing plan" ≠ video request).
        lower_query = query.lower()
        has_video_keywords = "video" in lower_query or "media" in lower_query
        has_strategy_keywords = "strategy" in lower_query or "analysis" in lower_query or "report" in lower_query or "recommendation" in lower_query
        
        if has_video_keywords and not has_strategy_keywords:
            session_id = f"video_only_{uuid.uuid4().hex}"
            logger.info("A2UI executor — detected video-only request, using fresh session: %s", session_id)

        # Get or create session
        session = await self._runner.session_service.get_session(
            app_name=self._runner.app_name,
            user_id=self._user_id,
            session_id=session_id,
        )
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._runner.app_name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )

        await updater.start_work()

        content = genai_types.Content(
            role="user", parts=[genai_types.Part.from_text(text=query)]
        )

        final_response_text = None

        try:
            async for event in self._runner.run_async(
                user_id=self._user_id,
                session_id=session.id,
                new_message=content,
            ):
                if event.is_final_response():
                    if (
                        event.content
                        and event.content.parts
                    ):
                        text_parts = [
                            p.text for p in event.content.parts if p.text
                        ]
                        if text_parts:
                            final_response_text = "\n".join(text_parts)
                            logger.info(
                                "A2UI executor — final response length: %d",
                                len(final_response_text),
                            )
        except Exception as exc:
            logger.exception("A2UI executor — runner failed")
            await updater.failed(
                message=a2a_utils.new_agent_text_message(
                    f"Task failed with error: {exc}"
                )
            )
            return

        if final_response_text is None:
            await updater.failed(
                message=a2a_utils.new_agent_text_message(
                    "No response generated."
                )
            )
            return

        # ---- Parse A2UI from the response text ----
        parts: list[a2a_types.Part] = []

        if _A2UI_DELIMITER not in final_response_text:
            # No A2UI — send as plain text
            logger.info("A2UI executor — no delimiter found, sending text")
            parts.append(
                a2a_types.Part(
                    root=a2a_types.TextPart(text=final_response_text)
                )
            )
        else:
            text_section, json_section = final_response_text.split(
                _A2UI_DELIMITER, 1
            )

            # Keep conversational text
            if text_section.strip():
                parts.append(
                    a2a_types.Part(
                        root=a2a_types.TextPart(text=text_section.strip())
                    )
                )

            # Parse and validate A2UI JSON
            json_str = (
                json_section.strip()
                .lstrip("```json")
                .lstrip("```")
                .rstrip("```")
                .strip()
            )

            if not json_str:
                json_str = "[]"

            try:
                a2ui_messages = json.loads(json_str)
                if not isinstance(a2ui_messages, list):
                    a2ui_messages = [a2ui_messages]

                for msg in a2ui_messages:
                    parts.append(
                        a2a_types.Part(
                            root=a2a_types.DataPart(
                                data=msg,
                                metadata={
                                    "mimeType": "application/json+a2ui",
                                },
                            )
                        )
                    )

                logger.info(
                    "A2UI executor — created %d DataParts from A2UI JSON",
                    len(a2ui_messages),
                )
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "A2UI executor — JSON parse failed: %s, sending raw text",
                    exc,
                )
                parts.append(
                    a2a_types.Part(
                        root=a2a_types.TextPart(text=final_response_text)
                    )
                )

        await updater.add_artifact(parts, name="response")
        await updater.complete()

    async def cancel(
        self,
        context: agent_execution.RequestContext,
        event_queue: a2a_events.EventQueue,
    ) -> None:
        raise a2a_errors.ServerError(error=a2a_types.UnsupportedOperationError())
