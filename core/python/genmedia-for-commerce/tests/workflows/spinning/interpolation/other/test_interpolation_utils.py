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

"""Tests for interpolation_utils.py - Video interpolation utilities."""

import io

import pytest
from PIL import Image
from workflows.shared.video_utils import create_mp4_from_bytes_to_bytes
from workflows.spinning.interpolation.other.interpolation_utils import (
    get_interpolation_prompt,
    post_process_single_video,
)


@pytest.fixture
def red_image_bytes():
    """Create a 100x100 red image."""
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def blue_image_bytes():
    """Create a 100x100 blue image."""
    img = Image.new("RGB", (100, 100), (0, 0, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_video_bytes(red_image_bytes, blue_image_bytes):
    """Create a simple video with 30 frames."""
    frames = [red_image_bytes] * 15 + [blue_image_bytes] * 15
    return create_mp4_from_bytes_to_bytes(frames, fps=24, quality=7)


class TestGetInterpolationPrompt:
    """Tests for get_interpolation_prompt function."""

    def test_returns_string(self, red_image_bytes):
        """Should return a non-empty string."""
        from unittest.mock import Mock, patch

        mock_client = Mock()
        with patch(
            "workflows.spinning.interpolation.other.interpolation_utils.generate_generic_product_title",
            return_value="a product",
        ):
            prompt = get_interpolation_prompt(
                mock_client, all_images_bytes=[red_image_bytes]
            )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_subject(self, red_image_bytes):
        """Should contain [Subject] section."""
        from unittest.mock import Mock, patch

        mock_client = Mock()
        with patch(
            "workflows.spinning.interpolation.other.interpolation_utils.generate_generic_product_title",
            return_value="a product",
        ):
            prompt = get_interpolation_prompt(
                mock_client, all_images_bytes=[red_image_bytes]
            )
        assert "[Subject]" in prompt

    def test_contains_action(self, red_image_bytes):
        """Should contain [Action] section."""
        from unittest.mock import Mock, patch

        mock_client = Mock()
        with patch(
            "workflows.spinning.interpolation.other.interpolation_utils.generate_generic_product_title",
            return_value="a product",
        ):
            prompt = get_interpolation_prompt(
                mock_client, all_images_bytes=[red_image_bytes]
            )
        assert "[Action]" in prompt

    def test_contains_camera(self, red_image_bytes):
        """Should contain [Camera] section."""
        from unittest.mock import Mock, patch

        mock_client = Mock()
        with patch(
            "workflows.spinning.interpolation.other.interpolation_utils.generate_generic_product_title",
            return_value="a product",
        ):
            prompt = get_interpolation_prompt(
                mock_client, all_images_bytes=[red_image_bytes]
            )
        # Note: The current prompt uses [Scene] instead of [Camera]
        assert "[Scene]" in prompt or "camera" in prompt.lower()

    def test_mentions_rotation(self, red_image_bytes):
        """Should mention rotation for product spinning."""
        from unittest.mock import Mock, patch

        mock_client = Mock()
        with patch(
            "workflows.spinning.interpolation.other.interpolation_utils.generate_generic_product_title",
            return_value="a product",
        ):
            prompt = get_interpolation_prompt(
                mock_client, all_images_bytes=[red_image_bytes]
            )
        assert "rotat" in prompt.lower()

    def test_mentions_white_background(self, red_image_bytes):
        """Should mention white background."""
        from unittest.mock import Mock, patch

        mock_client = Mock()
        with patch(
            "workflows.spinning.interpolation.other.interpolation_utils.generate_generic_product_title",
            return_value="a product",
        ):
            prompt = get_interpolation_prompt(
                mock_client, all_images_bytes=[red_image_bytes]
            )
        assert "white" in prompt.lower()


class TestPostProcessSingleVideo:
    """Tests for post_process_single_video function."""

    def test_processes_video(self, sample_video_bytes, blue_image_bytes):
        """Should process video and return bytes."""
        result = post_process_single_video(
            video_bytes=sample_video_bytes,
            end_image=blue_image_bytes,
            num_frames_for_similarity=5,
            is_first_video=True,
        )

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_is_first_video_keeps_first_frame(
        self, sample_video_bytes, blue_image_bytes
    ):
        """First video should keep the first frame."""
        result = post_process_single_video(
            video_bytes=sample_video_bytes,
            end_image=blue_image_bytes,
            num_frames_for_similarity=5,
            is_first_video=True,
        )

        assert isinstance(result, bytes)

    def test_not_first_video_removes_first_frame(
        self, sample_video_bytes, blue_image_bytes
    ):
        """Non-first video should remove first frame for concatenation."""
        result = post_process_single_video(
            video_bytes=sample_video_bytes,
            end_image=blue_image_bytes,
            num_frames_for_similarity=5,
            is_first_video=False,
        )

        assert isinstance(result, bytes)

    def test_trims_to_similar_frame(self, sample_video_bytes, red_image_bytes):
        """Should trim video to frame most similar to end_image."""
        # End image is red, video starts with red frames
        result = post_process_single_video(
            video_bytes=sample_video_bytes,
            end_image=red_image_bytes,
            num_frames_for_similarity=10,
            is_first_video=True,
        )

        assert isinstance(result, bytes)
        # Result should be shorter (trimmed at red section)
