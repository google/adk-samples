"""Tests for the consolidated Veo polling/extraction helpers (Phase 14b step 1).

These are the first tests ever covering the polling paths — previously
triplicated inline with zero coverage.
"""

import pytest

from app.tools import video_tools


class FakeVideo:
    def __init__(self, video_bytes=b"mp4-bytes"):
        self.video_bytes = video_bytes


class FakeGeneratedVideo:
    def __init__(self, video_bytes=b"mp4-bytes"):
        self.video = FakeVideo(video_bytes)


class FakeResult:
    def __init__(self, videos):
        self.generated_videos = videos


class FakeOperation:
    """Completes after `completes_after` polls; result set on completion."""

    def __init__(self, completes_after=0, result="ok"):
        self._polls_remaining = completes_after
        self._result_kind = result
        self.done = completes_after == 0

    @property
    def result(self):
        if self._result_kind == "ok":
            return FakeResult([FakeGeneratedVideo()])
        if self._result_kind == "empty":
            return FakeResult([])
        return None


class FakeClient:
    def __init__(self):
        self.polls = 0
        self.operations = None  # installed by make_client


def make_client(op: FakeOperation) -> FakeClient:
    client = FakeClient()

    class Ops:
        @staticmethod
        def get(operation):
            client.polls += 1
            op._polls_remaining -= 1
            if op._polls_remaining <= 0:
                op.done = True
            return op

    client.operations = Ops()
    return client


@pytest.fixture(autouse=True)
def instant_sleep(monkeypatch):
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(video_tools.asyncio, "sleep", _no_sleep)


class TestWaitForVeoOperation:
    @pytest.mark.asyncio
    async def test_success_after_polls(self):
        op = FakeOperation(completes_after=3)
        client = make_client(op)
        done = await video_tools._wait_for_veo_operation(client, op)
        assert done.done is True
        assert client.polls == 3
        assert done.result.generated_videos

    @pytest.mark.asyncio
    async def test_already_done_no_polls(self):
        op = FakeOperation(completes_after=0)
        client = make_client(op)
        done = await video_tools._wait_for_veo_operation(client, op)
        assert client.polls == 0
        assert done is op

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        op = FakeOperation(completes_after=10_000)
        client = make_client(op)
        with pytest.raises(TimeoutError, match="timed out after 600"):
            await video_tools._wait_for_veo_operation(client, op)
        # 600s budget / 20s interval = 30 polls exactly
        assert client.polls == 30

    @pytest.mark.asyncio
    async def test_empty_result_raises_valueerror(self):
        op = FakeOperation(completes_after=1, result="empty")
        client = make_client(op)
        with pytest.raises(ValueError, match="no result"):
            await video_tools._wait_for_veo_operation(client, op)

    @pytest.mark.asyncio
    async def test_none_result_raises_valueerror(self):
        op = FakeOperation(completes_after=1, result="none")
        client = make_client(op)
        with pytest.raises(ValueError, match="no result"):
            await video_tools._wait_for_veo_operation(client, op)


class TestExtractVideoBytes:
    def test_vertex_inline_bytes(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
        out = video_tools._extract_video_bytes(
            object(), FakeGeneratedVideo(b"abc")
        )
        assert out == b"abc"

    def test_vertex_empty_bytes_raises(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
        with pytest.raises(ValueError, match="No video_bytes"):
            video_tools._extract_video_bytes(object(), FakeGeneratedVideo(b""))


class TestGenerateVideoAdTimeoutSemantics:
    """generate_video_ad must keep its DB-update + error-dict timeout behavior
    (it maps the helper's exceptions at the call site instead of raising)."""

    @pytest.mark.asyncio
    async def test_timeout_marks_ad_failed_and_returns_error(
        self, fresh_test_db, monkeypatch
    ):
        import io as _io

        from PIL import Image as PILImage

        # Minimal valid PNG for the PIL sniff in generate_video_ad
        buf = _io.BytesIO()
        PILImage.new("RGB", (4, 4)).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        from app.database.db import get_db_cursor

        with get_db_cursor() as cursor:
            cursor.execute(
                "INSERT INTO campaigns (name, city, state, category) VALUES (?, ?, ?, ?)",
                ("veo-timeout-test", "Los Angeles", "CA", "holiday"),
            )
            campaign_id = cursor.lastrowid
            cursor.execute(
                "INSERT INTO campaign_images (campaign_id, image_path, metadata) VALUES (?, ?, ?)",
                (campaign_id, "veo-timeout-test.png", "{}"),
            )

        monkeypatch.setattr(
            video_tools.storage, "get_storage_mode", lambda: "local"
        )
        monkeypatch.setattr(
            video_tools.storage, "get_image_path", lambda f: f"/tmp/{f}"
        )
        monkeypatch.setattr(video_tools.storage, "image_exists", lambda f: True)
        monkeypatch.setattr(
            video_tools.storage, "read_image", lambda f: png_bytes
        )

        never_done = FakeOperation(completes_after=10_000)
        fake_client = make_client(never_done)
        fake_client.models = type(
            "M", (), {"generate_videos": staticmethod(lambda **kw: never_done)}
        )()
        monkeypatch.setattr(video_tools.genai, "Client", lambda: fake_client)

        result = await video_tools.generate_video_ad(
            campaign_id=campaign_id,
            custom_prompt="test prompt",
            duration_seconds=4,
        )

        assert result["status"] == "error"
        assert "timed out" in result["message"]
        with get_db_cursor() as cursor:
            cursor.execute(
                "SELECT status FROM campaign_ads WHERE id = ?",
                (result["ad_id"],),
            )
            assert cursor.fetchone()["status"] == "failed"
