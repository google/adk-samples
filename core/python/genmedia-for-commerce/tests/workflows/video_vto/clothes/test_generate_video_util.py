# Copyright 2025 Google LLC
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

"""Tests for generate_video_util.py - Clothes Video VTO R2V pipeline."""

import io
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from workflows.video_vto.clothes.generate_video_util import (
    _create_three_framings,
    run_r2v_pipeline,
)


@pytest.fixture
def full_body_image_bytes():
    """Create a 400x800 full-body image (portrait)."""
    img = Image.new("RGB", (400, 800), color=(200, 150, 100))
    # Add a distinct region at the top for the "face"
    for x in range(150, 250):
        for y in range(50, 150):
            img.putpixel((x, y), (255, 200, 180))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestCreateThreeFramings:
    """Tests for _create_three_framings function."""

    @patch("workflows.video_vto.clothes.generate_video_util.save_debug_image")
    @patch("workflows.video_vto.clothes.generate_video_util.crop_face")
    def test_returns_three_images(
        self, mock_crop_face, mock_save, full_body_image_bytes
    ):
        """Should return a tuple of three PNG byte strings."""
        mock_crop_face.return_value = None  # Force fallback

        lower, upper, face = _create_three_framings(full_body_image_bytes)

        assert isinstance(lower, bytes)
        assert isinstance(upper, bytes)
        assert isinstance(face, bytes)

    @patch("workflows.video_vto.clothes.generate_video_util.save_debug_image")
    @patch("workflows.video_vto.clothes.generate_video_util.crop_face")
    def test_lower_body_is_bottom_60_percent(
        self, mock_crop_face, mock_save, full_body_image_bytes
    ):
        """Lower body crop should be the bottom 60% of the image."""
        mock_crop_face.return_value = None

        lower, _, _ = _create_three_framings(full_body_image_bytes)

        lower_img = Image.open(io.BytesIO(lower))
        # Original is 400x800; bottom 60% => height = 800 * 0.6 = 480
        assert lower_img.size == (400, 480)

    @patch("workflows.video_vto.clothes.generate_video_util.save_debug_image")
    @patch("workflows.video_vto.clothes.generate_video_util.crop_face")
    def test_upper_body_is_top_40_percent(
        self, mock_crop_face, mock_save, full_body_image_bytes
    ):
        """Upper body crop should be the top 40% of the image."""
        mock_crop_face.return_value = None

        _, upper, _ = _create_three_framings(full_body_image_bytes)

        upper_img = Image.open(io.BytesIO(upper))
        # Original is 400x800; top 40% => height = 800 * 0.4 = 320
        assert upper_img.size == (400, 320)

    @patch("workflows.video_vto.clothes.generate_video_util.save_debug_image")
    @patch("workflows.video_vto.clothes.generate_video_util.crop_face")
    def test_face_fallback_when_no_face_detected(
        self, mock_crop_face, mock_save, full_body_image_bytes
    ):
        """When crop_face returns None, face crop should be the top 30%."""
        mock_crop_face.return_value = None

        _, _, face = _create_three_framings(full_body_image_bytes)

        face_img = Image.open(io.BytesIO(face))
        # Fallback: top 30% => height = 800 * 0.3 = 240
        assert face_img.size == (400, 240)

    @patch("workflows.video_vto.clothes.generate_video_util.save_debug_image")
    @patch("workflows.video_vto.clothes.generate_video_util.crop_face")
    def test_face_uses_crop_face_when_available(
        self, mock_crop_face, mock_save
    ):
        """When crop_face returns bytes, those should be used for the face."""
        # Create a small face image to return from crop_face
        face_img = Image.new("RGB", (80, 80), color=(255, 200, 180))
        face_buf = io.BytesIO()
        face_img.save(face_buf, format="PNG")
        face_bytes = face_buf.getvalue()
        mock_crop_face.return_value = face_bytes

        # Create input image
        img = Image.new("RGB", (400, 800), color=(200, 150, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        _, _, face = _create_three_framings(buf.getvalue())

        face_result = Image.open(io.BytesIO(face))
        assert face_result.size == (80, 80)


class TestRunR2vPipeline:
    """Tests for run_r2v_pipeline function."""

    @patch("workflows.video_vto.clothes.generate_video_util.save_debug_image")
    @patch(
        "workflows.video_vto.clothes.generate_video_util.create_canvas",
        side_effect=lambda b, **kw: b,
    )
    @patch(
        "workflows.video_vto.clothes.generate_video_util.crop_face",
        return_value=None,
    )
    @patch("workflows.video_vto.clothes.generate_video_util.generate_veo_r2v")
    def test_returns_expected_keys(
        self, mock_veo, mock_crop, mock_canvas, mock_save, full_body_image_bytes
    ):
        """Result dict should contain videos, first_frame, last_frame, aborted."""
        mock_veo.return_value = b"fake_video_bytes"
        mock_client = Mock()

        result = run_r2v_pipeline(
            veo_client=mock_client,
            model_image_bytes=full_body_image_bytes,
            prompt="test prompt",
            number_of_videos=1,
        )

        assert "videos" in result
        assert "first_frame" in result
        assert "last_frame" in result
        assert "aborted" in result
        assert result["aborted"] is False
        assert len(result["videos"]) == 1

    @patch("workflows.video_vto.clothes.generate_video_util.save_debug_image")
    @patch(
        "workflows.video_vto.clothes.generate_video_util.create_canvas",
        side_effect=lambda b, **kw: b,
    )
    @patch(
        "workflows.video_vto.clothes.generate_video_util.crop_face",
        return_value=None,
    )
    @patch("workflows.video_vto.clothes.generate_video_util.generate_veo_r2v")
    def test_abort_when_first_clip_check_fails(
        self, mock_veo, mock_crop, mock_canvas, mock_save, full_body_image_bytes
    ):
        """When first_clip_check returns False, result should be aborted with no videos."""
        mock_veo.return_value = b"fake_video_bytes"
        mock_client = Mock()

        result = run_r2v_pipeline(
            veo_client=mock_client,
            model_image_bytes=full_body_image_bytes,
            prompt="test prompt",
            number_of_videos=2,
            first_clip_check=lambda _: False,
        )

        assert result["aborted"] is True
        assert result["videos"] == []
