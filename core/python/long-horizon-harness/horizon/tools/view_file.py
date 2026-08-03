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

"""``view_file(path)`` — inject a workspace file's bytes into the agent's
own next-turn LLM context as a multimodal ``Part``.

Companion to ``artifact(action='save')`` (which surfaces files to the UI):
``view_file`` is for the agent itself. Use it for binaries Gemini natively
parses (PDF, images, audio) instead of shelling out to extraction tools.

Mirrors ADK's ``LoadArtifactsTool``: a single ``BaseTool`` subclass with
``run_async`` (does the save) and ``process_llm_request`` (injects the
Part on the next turn). State key ``_pending_view_files`` carries the
filename(s) between the two callbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.adk.tools.base_tool import BaseTool
from google.genai import types as genai_types
from typing_extensions import override

from horizon.environment_context import active_environment
from horizon.models.media import b64_len
from horizon.models.registry import model_capabilities
from horizon.models.selector import resolve_model_name
from horizon.tools.file_ops import guess_mime_from_bytes
from horizon.workspace_window import resolve_in_window, window_dirs

if TYPE_CHECKING:
    from google.adk.models import LlmRequest
    from google.adk.tools.tool_context import ToolContext


_PENDING_STATE_KEY = "_pending_view_files"


class ViewFileTool(BaseTool):
    """Load a workspace file into the agent's own LLM context as a
    multimodal Part for the next turn."""

    def __init__(self) -> None:
        super().__init__(
            name="view_file",
            description=(
                "Load a workspace file directly into your own context as a "
                "multimodal input — use this ONLY when you must analyze the "
                "content yourself (read a chart, transcribe audio, OCR a scan). "
                "The bytes are loaded into your context and cost tokens every "
                "turn they persist, so do NOT use it just to 'confirm quality' "
                "of a file you generated for the user — to show the user a file "
                "without spending your own tokens, use `artifact(save)` "
                "instead. Images, PDFs, audio, and video all work with Gemini. "
                "After calling, the file appears as inline data in your NEXT "
                "turn. For plain-text files use `read_file` instead."
            ),
        )

    def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "path": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description=(
                            "Path of the file to load. Relative paths resolve "
                            "under your current workspace focus; prefix with `/` "
                            "to reach the whole workspace."
                        ),
                    ),
                },
                required=["path"],
            ),
        )

    @override
    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any]:
        path = args.get("path")
        if not path:
            return {"success": False, "error": "missing required arg 'path'"}

        target, err = resolve_in_window(
            path,
            active_environment().working_dir.resolve(),
            window_dirs(getattr(tool_context, "state", None)),
        )
        if err is not None:
            return {"success": False, "error": err}
        assert target is not None

        try:
            data = await active_environment().read_file(target)
        except FileNotFoundError:
            return {"success": False, "error": f"File not found: {path}"}
        except OSError as exc:
            return {"success": False, "error": f"Failed to read {path}: {exc}"}

        filename = target.name
        mime_type = guess_mime_from_bytes(filename, data)
        part = genai_types.Part(
            inline_data=genai_types.Blob(
                data=data,
                mime_type=mime_type,
                display_name=filename,
            )
        )
        version = await tool_context.save_artifact(
            filename=filename, artifact=part
        )
        # Populating artifact_delta is what makes the A2A interceptor surface
        # the FilePart in the chat — so the user sees (and can download) what
        # the agent just loaded into its own context.
        tool_context.actions.artifact_delta[filename] = version

        result: dict[str, Any] = {
            "success": True,
            "filename": filename,
            "mime_type": mime_type,
            "size_bytes": len(data),
        }

        model_name = resolve_model_name(getattr(tool_context, "state", None))
        caps = model_capabilities(model_name)
        if not caps.can_view_mime(mime_type):
            # This model has no audio/video/binary input; injecting it would be
            # dropped downstream. Surface to the UI but tell the agent it
            # won't reach its context.
            result["model_can_view"] = False
            result["note"] = (
                f"Surfaced to the UI but NOT loaded into your context: the "
                f"active model ({model_name}) cannot view {mime_type}. "
                f"Use read_file for text."
            )
            return result

        if (
            caps.max_image_bytes is not None
            and mime_type.startswith("image/")
            and b64_len(len(data)) > caps.max_image_bytes
        ):
            # This model rejects an inline image over its base64 cap. Surface to
            # the UI but don't inject — tell the agent to shrink it and retry.
            cap_mb = caps.max_image_bytes / (1024 * 1024)
            result["too_large"] = True
            result["note"] = (
                f"Surfaced to the UI but NOT loaded into your context: this "
                f"image is {len(data)} bytes, over the active model's "
                f"{cap_mb:.0f} MB per-image limit. Resize or compress it under "
                f"{cap_mb:.0f} MB (e.g. in the sandbox), then view_file it again."
            )
            return result

        pending = list(tool_context.state.get(_PENDING_STATE_KEY, []))
        pending.append(filename)
        tool_context.state[_PENDING_STATE_KEY] = pending
        return result

    @override
    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: LlmRequest
    ) -> None:
        await super().process_llm_request(
            tool_context=tool_context, llm_request=llm_request
        )

        pending = tool_context.state.get(_PENDING_STATE_KEY)
        if not pending:
            return
        # Clear regardless of load outcome — never re-inject the same file.
        tool_context.state[_PENDING_STATE_KEY] = []

        for filename in pending:
            part = await tool_context.load_artifact(filename=filename)
            if part is None:
                continue
            llm_request.contents.append(
                genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part.from_text(
                            text=f"Contents of {filename}:"
                        ),
                        part,
                    ],
                )
            )


__all__ = ["ViewFileTool"]
