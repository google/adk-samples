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

"""``ViewFileTool`` — inject workspace file bytes into the agent's own
next-turn LLM context as a multimodal ``Part``.

The tool has two halves that pair across a turn boundary:

* ``run_async`` (LLM invokes it): validate path, read bytes through the
  env interface, ``save_artifact``, set ``actions.artifact_delta``, stash the
  basename in ``state['_pending_view_files']``.
* ``process_llm_request`` (runner fires before next LLM call): pop the
  state list, ``load_artifact``, append the Part to
  ``llm_request.contents``.

The state round-trip is the only way to carry bytes across the turn —
session state on Vertex can't hold raw ``bytes`` (JSON-serialized).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from google.adk.models import LlmRequest
from google.genai import types as genai_types

from horizon.environment import LocalEnvironment


class _FakeActions:
    """Mirror of ``EventActions.artifact_delta``."""

    def __init__(self) -> None:
        self.artifact_delta: dict[str, int] = {}


class _FakeToolContext:
    """Minimal stand-in for ``ToolContext``: artifact store + state dict +
    actions. ``state`` is a plain dict — the production ``State`` object's
    delta tracking is not needed for unit assertions."""

    def __init__(self) -> None:
        self._store: dict[str, list[genai_types.Part]] = {}
        self.actions = _FakeActions()
        self.state: dict[str, Any] = {}

    async def save_artifact(
        self,
        filename: str,
        artifact: genai_types.Part,
        custom_metadata: dict[str, Any] | None = None,
    ) -> int:
        versions = self._store.setdefault(filename, [])
        versions.append(artifact)
        return len(versions) - 1

    async def load_artifact(
        self, filename: str, version: int | None = None
    ) -> genai_types.Part | None:
        versions = self._store.get(filename)
        if not versions:
            return None
        if version is None:
            return versions[-1]
        return versions[version]

    async def list_artifacts(self) -> list[str]:
        return list(self._store.keys())


@pytest.fixture
def fake_ctx() -> _FakeToolContext:
    return _FakeToolContext()


@pytest.fixture
def local_env(tmp_path: Path) -> LocalEnvironment:
    """LocalEnvironment rooted at tmp_path, installed in the ContextVar
    so the tool has real bytes to read through the env contract."""
    from horizon.environment_context import set_active_environment

    env = LocalEnvironment(working_dir=tmp_path)
    set_active_environment(env)
    return env


pytestmark = pytest.mark.asyncio


# =============================================================================
# run_async — load workspace file into artifact store + stash in state
# =============================================================================


async def test_view_file_happy_path(
    local_env: LocalEnvironment, fake_ctx: _FakeToolContext
) -> None:
    """Happy path: reads bytes through env, saves an artifact with correct
    mime/display_name, records the basename in pending state, and populates
    artifact_delta so the file also reaches the UI."""
    from horizon.tools.view_file import ViewFileTool

    payload = b"%PDF-1.4\nfake-pdf-bytes\n%%EOF"
    (local_env.working_dir / "doc.pdf").write_bytes(payload)

    tool = ViewFileTool()
    result = await tool.run_async(
        args={"path": "doc.pdf"}, tool_context=fake_ctx
    )

    assert result == {
        "success": True,
        "filename": "doc.pdf",
        "mime_type": "application/pdf",
        "size_bytes": len(payload),
    }
    assert fake_ctx.state["_pending_view_files"] == ["doc.pdf"]
    assert fake_ctx.actions.artifact_delta == {"doc.pdf": 0}

    saved = await fake_ctx.load_artifact("doc.pdf")
    assert saved is not None
    assert saved.inline_data is not None
    assert saved.inline_data.data == payload
    assert saved.inline_data.mime_type == "application/pdf"
    assert saved.inline_data.display_name == "doc.pdf"


async def test_view_file_rejects_path_traversal(
    local_env: LocalEnvironment, fake_ctx: _FakeToolContext
) -> None:
    """Path escaping the working_dir is refused. No state/artifact effects."""
    from horizon.tools.view_file import ViewFileTool

    tool = ViewFileTool()
    result = await tool.run_async(
        args={"path": "../../../etc/passwd"}, tool_context=fake_ctx
    )

    assert result["success"] is False
    assert "error" in result
    assert "_pending_view_files" not in fake_ctx.state
    assert fake_ctx.actions.artifact_delta == {}
    assert await fake_ctx.list_artifacts() == []


async def test_view_file_missing_file(
    local_env: LocalEnvironment, fake_ctx: _FakeToolContext
) -> None:
    """Non-existent path returns structured error; no side effects."""
    from horizon.tools.view_file import ViewFileTool

    tool = ViewFileTool()
    result = await tool.run_async(
        args={"path": "nope.pdf"}, tool_context=fake_ctx
    )

    assert result["success"] is False
    assert "nope.pdf" in result["error"]
    assert "_pending_view_files" not in fake_ctx.state
    assert fake_ctx.actions.artifact_delta == {}
    assert await fake_ctx.list_artifacts() == []


async def test_view_file_unknown_mime_defaults_to_octet_stream(
    local_env: LocalEnvironment, fake_ctx: _FakeToolContext
) -> None:
    """Files with no recognized extension fall back to application/octet-stream."""
    from horizon.tools.view_file import ViewFileTool

    (local_env.working_dir / "blob").write_bytes(b"\x00\xff\x80\x01")

    tool = ViewFileTool()
    result = await tool.run_async(args={"path": "blob"}, tool_context=fake_ctx)

    assert result["success"] is True
    assert result["mime_type"] == "application/octet-stream"


async def test_view_file_mime_from_bytes_not_extension(
    local_env: LocalEnvironment, fake_ctx: _FakeToolContext
) -> None:
    """A file named .png whose bytes are actually JPEG must report image/jpeg
    (sniffed from magic bytes) so the model receives a correctly-typed blob."""
    from horizon.tools.view_file import ViewFileTool

    jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 8
    (local_env.working_dir / "shot.png").write_bytes(jpeg)

    tool = ViewFileTool()
    result = await tool.run_async(
        args={"path": "shot.png"}, tool_context=fake_ctx
    )

    assert result["success"] is True
    assert result["mime_type"] == "image/jpeg"

    saved = await fake_ctx.load_artifact("shot.png")
    assert saved is not None
    assert saved.inline_data is not None
    assert saved.inline_data.mime_type == "image/jpeg"


async def test_view_file_appends_when_called_twice(
    local_env: LocalEnvironment, fake_ctx: _FakeToolContext
) -> None:
    """Two view_file calls in one turn accumulate in the pending state list
    so process_llm_request injects both next turn."""
    from horizon.tools.view_file import ViewFileTool

    (local_env.working_dir / "a.png").write_bytes(b"\x89PNG\r\nfake-a")
    (local_env.working_dir / "b.png").write_bytes(b"\x89PNG\r\nfake-b")

    tool = ViewFileTool()
    await tool.run_async(args={"path": "a.png"}, tool_context=fake_ctx)
    await tool.run_async(args={"path": "b.png"}, tool_context=fake_ctx)

    assert fake_ctx.state["_pending_view_files"] == ["a.png", "b.png"]


# =============================================================================
# run_async — model-aware media gating (the ModelCapabilities boundary)
# =============================================================================
#
# Both registered models are Gemini (view everything, no image cap), so the
# gate never fires in production. These tests drive it via a synthetic
# restricted ModelCapabilities to prove the boundary still works for a future
# model with narrower media support.


def _restricted_caps(*, max_image_bytes: int | None = None):
    from horizon.models.capabilities import ModelCapabilities

    def _image_or_pdf(mime: str | None) -> bool:
        base = (mime or "").split(";")[0].strip().lower()
        return base.startswith("image/") or base == "application/pdf"

    return ModelCapabilities(
        max_image_bytes=max_image_bytes,
        can_view_mime=_image_or_pdf,
        prepare_contents=None,
    )


async def test_view_file_audio_not_injected_when_model_cannot_view(
    local_env: LocalEnvironment, fake_ctx: _FakeToolContext, monkeypatch
) -> None:
    """When the active model can't ingest a mime type, the file is surfaced to
    the UI but NOT queued for context injection — the result flags it."""
    from horizon.tools import view_file as vf

    monkeypatch.setattr(vf, "model_capabilities", lambda _n: _restricted_caps())
    (local_env.working_dir / "clip.mp3").write_bytes(b"ID3\x03\x00audio")

    result = await vf.ViewFileTool().run_async(
        args={"path": "clip.mp3"}, tool_context=fake_ctx
    )
    assert result["success"] is True
    assert result["model_can_view"] is False
    assert fake_ctx.state.get("_pending_view_files", []) == []
    assert fake_ctx.actions.artifact_delta == {"clip.mp3": 0}


async def test_view_file_audio_injected_on_gemini(
    local_env: LocalEnvironment, fake_ctx: _FakeToolContext
) -> None:
    """Gemini ingests audio natively, so the audio file IS queued for
    injection — the media gate must not fire for Gemini sessions."""
    from horizon.tools.view_file import ViewFileTool

    fake_ctx.state["selected_model"] = "gemini-3.6-flash"
    (local_env.working_dir / "clip.mp3").write_bytes(b"ID3\x03\x00audio")

    result = await ViewFileTool().run_async(
        args={"path": "clip.mp3"}, tool_context=fake_ctx
    )
    assert result["success"] is True
    assert "model_can_view" not in result
    assert fake_ctx.state["_pending_view_files"] == ["clip.mp3"]


_IMG_CAP_BYTES = 5 * 1024 * 1024


def _oversized_png() -> bytes:
    """A PNG whose base64 length exceeds _IMG_CAP_BYTES. Magic bytes make it
    sniff as image/png; the body is padding (the guard is a pure size check)."""
    raw_budget = _IMG_CAP_BYTES * 3 // 4
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * (raw_budget + 1024)


async def test_view_file_oversized_image_not_injected_when_capped(
    local_env: LocalEnvironment, fake_ctx: _FakeToolContext, monkeypatch
) -> None:
    """An image over the active model's per-image cap is surfaced to the UI but
    NOT queued for injection; the result flags it so the agent can downscale."""
    from horizon.tools import view_file as vf

    monkeypatch.setattr(
        vf,
        "model_capabilities",
        lambda _n: _restricted_caps(max_image_bytes=_IMG_CAP_BYTES),
    )
    (local_env.working_dir / "big.png").write_bytes(_oversized_png())

    result = await vf.ViewFileTool().run_async(
        args={"path": "big.png"}, tool_context=fake_ctx
    )
    assert result["success"] is True
    assert result["too_large"] is True
    assert fake_ctx.state.get("_pending_view_files", []) == []
    assert fake_ctx.actions.artifact_delta == {"big.png": 0}


async def test_view_file_oversized_image_injected_on_gemini(
    local_env: LocalEnvironment, fake_ctx: _FakeToolContext
) -> None:
    """Gemini has no per-image byte cap (max_image_bytes=None), so the same
    oversized file IS queued for injection."""
    from horizon.tools.view_file import ViewFileTool

    fake_ctx.state["selected_model"] = "gemini-3.6-flash"
    (local_env.working_dir / "big.png").write_bytes(_oversized_png())

    result = await ViewFileTool().run_async(
        args={"path": "big.png"}, tool_context=fake_ctx
    )
    assert result["success"] is True
    assert "too_large" not in result
    assert fake_ctx.state["_pending_view_files"] == ["big.png"]


# =============================================================================
# process_llm_request — pull pending bytes back into llm_request.contents
# =============================================================================


async def test_process_llm_request_injects_pending_files(
    fake_ctx: _FakeToolContext,
) -> None:
    """Pending files in state get loaded from the artifact store and appended
    to llm_request.contents as user-role Content; state is cleared after."""
    from horizon.tools.view_file import ViewFileTool

    payload = b"%PDF-1.4 contents"
    part = genai_types.Part(
        inline_data=genai_types.Blob(
            data=payload, mime_type="application/pdf", display_name="doc.pdf"
        )
    )
    await fake_ctx.save_artifact("doc.pdf", artifact=part)
    fake_ctx.state["_pending_view_files"] = ["doc.pdf"]

    request = LlmRequest(model="gemini-3.6-flash")
    before = len(request.contents)

    tool = ViewFileTool()
    await tool.process_llm_request(tool_context=fake_ctx, llm_request=request)

    assert len(request.contents) == before + 1
    injected = request.contents[-1]
    assert injected.role == "user"
    assert injected.parts is not None
    assert len(injected.parts) == 2
    assert injected.parts[0].text == "Contents of doc.pdf:"
    assert injected.parts[1].inline_data is not None
    assert injected.parts[1].inline_data.data == payload
    assert fake_ctx.state["_pending_view_files"] == []


async def test_process_llm_request_noop_when_empty(
    fake_ctx: _FakeToolContext,
) -> None:
    """No pending files → contents are not mutated (beyond the base-class
    function-declaration registration)."""
    from horizon.tools.view_file import ViewFileTool

    request = LlmRequest(model="gemini-3.6-flash")
    before = len(request.contents)

    tool = ViewFileTool()
    await tool.process_llm_request(tool_context=fake_ctx, llm_request=request)

    assert len(request.contents) == before


async def test_process_llm_request_skips_evicted_artifacts(
    fake_ctx: _FakeToolContext,
) -> None:
    """If an artifact was evicted between turns, skip it silently rather
    than crashing. State is still cleared."""
    from horizon.tools.view_file import ViewFileTool

    fake_ctx.state["_pending_view_files"] = ["gone.pdf"]

    request = LlmRequest(model="gemini-3.6-flash")
    before = len(request.contents)

    tool = ViewFileTool()
    await tool.process_llm_request(tool_context=fake_ctx, llm_request=request)

    assert len(request.contents) == before
    assert fake_ctx.state["_pending_view_files"] == []
