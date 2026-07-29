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

"""Tests for shared/veo_utils.py - Veo video generation with mocked client."""

import io
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from workflows.shared.veo_utils import (
    VeoEmptyResultError,
    generate_veo,
    generate_veo_r2v,
)


@pytest.fixture
def sample_image_bytes():
    """Create a sample PNG image as bytes."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_video_bytes():
    """Create sample video bytes (fake MP4 content)."""
    return b"fake_video_content_mp4"


@pytest.fixture
def mock_veo_client(sample_video_bytes):
    """Create a mock Veo client that returns successful video generation."""
    client = Mock()

    # Create mock operation that is already done
    mock_operation = Mock()
    mock_operation.done = True
    mock_operation.response = True

    # Mock video result
    mock_video = Mock()
    mock_video.video.video_bytes = sample_video_bytes
    mock_operation.result.generated_videos = [mock_video]

    # Mock models.generate_videos
    client.models.generate_videos.return_value = mock_operation

    return client


@pytest.fixture
def mock_veo_client_pending(sample_video_bytes):
    """Create a mock Veo client that starts pending then completes."""
    client = Mock()

    # Create operations that simulate pending -> done transition
    pending_op = Mock()
    pending_op.done = False

    done_op = Mock()
    done_op.done = True
    done_op.response = True
    mock_video = Mock()
    mock_video.video.video_bytes = sample_video_bytes
    done_op.result.generated_videos = [mock_video]

    # First call returns pending, subsequent call returns done
    client.models.generate_videos.return_value = pending_op
    client.operations.get.return_value = done_op

    return client


@pytest.fixture
def mock_veo_client_empty():
    """Create a mock Veo client that returns empty result."""
    client = Mock()

    mock_operation = Mock()
    mock_operation.done = True
    mock_operation.response = True
    mock_operation.result.generated_videos = []

    client.models.generate_videos.return_value = mock_operation

    return client


class TestGenerateVeo:
    """Tests for generate_veo function."""

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_generates_video_from_image(
        self, mock_sleep, mock_veo_client, sample_image_bytes, sample_video_bytes
    ):
        """Should generate video from a single image."""
        result = generate_veo(
            client=mock_veo_client,
            image=sample_image_bytes,
            prompt="A product rotating smoothly",
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == sample_video_bytes
        mock_veo_client.models.generate_videos.assert_called_once()

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_generates_video_with_last_frame(
        self, mock_sleep, mock_veo_client, sample_image_bytes, sample_video_bytes
    ):
        """Should generate video with interpolation (start and end frames)."""
        # Create a different end frame
        end_img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        end_buffer = io.BytesIO()
        end_img.save(end_buffer, format="PNG")
        end_frame = end_buffer.getvalue()

        result = generate_veo(
            client=mock_veo_client,
            image=sample_image_bytes,
            prompt="Smooth transition",
            last_frame=end_frame,
        )

        assert len(result) == 1
        assert result[0] == sample_video_bytes

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_respects_duration_parameter(
        self, mock_sleep, mock_veo_client, sample_image_bytes
    ):
        """Should pass duration to config."""
        generate_veo(
            client=mock_veo_client,
            image=sample_image_bytes,
            prompt="Test prompt",
            duration=5,
        )

        call_kwargs = mock_veo_client.models.generate_videos.call_args
        config = call_kwargs.kwargs["config"]
        assert config.duration_seconds == 5

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_respects_aspect_ratio(
        self, mock_sleep, mock_veo_client, sample_image_bytes
    ):
        """Should pass aspect ratio to config."""
        generate_veo(
            client=mock_veo_client,
            image=sample_image_bytes,
            prompt="Test prompt",
            aspect_ratio="9:16",
        )

        call_kwargs = mock_veo_client.models.generate_videos.call_args
        config = call_kwargs.kwargs["config"]
        assert config.aspect_ratio == "9:16"

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_handles_multiple_videos(
        self, mock_sleep, sample_image_bytes, sample_video_bytes
    ):
        """Should return multiple videos when requested."""
        client = Mock()
        mock_operation = Mock()
        mock_operation.done = True
        mock_operation.response = True

        # Create 3 mock videos
        mock_videos = []
        for i in range(3):
            mock_video = Mock()
            mock_video.video.video_bytes = f"video_{i}".encode()
            mock_videos.append(mock_video)
        mock_operation.result.generated_videos = mock_videos

        client.models.generate_videos.return_value = mock_operation

        result = generate_veo(
            client=client,
            image=sample_image_bytes,
            prompt="Test prompt",
            number_of_videos=3,
        )

        assert len(result) == 3

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_waits_for_operation_completion(
        self, mock_sleep, mock_veo_client_pending, sample_image_bytes
    ):
        """Should poll until operation is done."""
        result = generate_veo(
            client=mock_veo_client_pending,
            image=sample_image_bytes,
            prompt="Test prompt",
        )

        assert len(result) == 1
        mock_veo_client_pending.operations.get.assert_called()
        mock_sleep.assert_called()

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_raises_on_empty_result(
        self, mock_sleep, mock_veo_client_empty, sample_image_bytes
    ):
        """Should raise VeoEmptyResultError when no videos generated."""
        # Disable retry decorator for this test
        with patch.object(generate_veo, "__wrapped__", generate_veo.__wrapped__):
            with pytest.raises(VeoEmptyResultError, match="empty result"):
                generate_veo.__wrapped__(
                    client=mock_veo_client_empty,
                    image=sample_image_bytes,
                    prompt="Test prompt",
                )

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_person_generation_parameter(
        self, mock_sleep, mock_veo_client, sample_image_bytes
    ):
        """Should pass person_generation to config when specified."""
        generate_veo(
            client=mock_veo_client,
            image=sample_image_bytes,
            prompt="Test prompt",
            person_generation="allow_adult",
        )

        call_kwargs = mock_veo_client.models.generate_videos.call_args
        config = call_kwargs.kwargs["config"]
        assert config.person_generation == "allow_adult"

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_enhance_prompt_parameter(
        self, mock_sleep, mock_veo_client, sample_image_bytes
    ):
        """Should pass enhance_prompt to config when specified."""
        generate_veo(
            client=mock_veo_client,
            image=sample_image_bytes,
            prompt="Test prompt",
            enhance_prompt=True,
        )

        call_kwargs = mock_veo_client.models.generate_videos.call_args
        config = call_kwargs.kwargs["config"]
        assert config.enhance_prompt

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_generate_audio_parameter(
        self, mock_sleep, mock_veo_client, sample_image_bytes
    ):
        """Should pass generate_audio to config."""
        generate_veo(
            client=mock_veo_client,
            image=sample_image_bytes,
            prompt="Test prompt",
            generate_audio=True,
        )

        call_kwargs = mock_veo_client.models.generate_videos.call_args
        config = call_kwargs.kwargs["config"]
        assert config.generate_audio


class TestGenerateVeoR2V:
    """Tests for generate_veo_r2v function (Reference-to-Video)."""

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_generates_video_from_references(
        self, mock_sleep, mock_veo_client, sample_image_bytes, sample_video_bytes
    ):
        """Should generate video from reference images."""
        reference_images = [sample_image_bytes, sample_image_bytes, sample_image_bytes]

        result = generate_veo_r2v(
            client=mock_veo_client,
            reference_images=reference_images,
            prompt="A product rotating on white background",
        )

        assert result == sample_video_bytes
        mock_veo_client.models.generate_videos.assert_called_once()

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_uses_asset_reference_type(
        self, mock_sleep, mock_veo_client, sample_image_bytes
    ):
        """Should use 'asset' as reference type by default."""
        generate_veo_r2v(
            client=mock_veo_client,
            reference_images=[sample_image_bytes],
            prompt="Test prompt",
            reference_type="asset",
        )

        call_kwargs = mock_veo_client.models.generate_videos.call_args
        config = call_kwargs.kwargs["config"]
        # Check reference_images in config
        assert len(config.reference_images) == 1

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_supports_style_reference_type(
        self, mock_sleep, mock_veo_client, sample_image_bytes
    ):
        """Should support 'style' reference type."""
        generate_veo_r2v(
            client=mock_veo_client,
            reference_images=[sample_image_bytes],
            prompt="Test prompt",
            reference_type="style",
        )

        mock_veo_client.models.generate_videos.assert_called_once()

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_handles_single_reference(
        self, mock_sleep, mock_veo_client, sample_image_bytes, sample_video_bytes
    ):
        """Should handle single reference image."""
        result = generate_veo_r2v(
            client=mock_veo_client,
            reference_images=[sample_image_bytes],
            prompt="Test prompt",
        )

        assert result == sample_video_bytes

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_handles_multiple_references(
        self, mock_sleep, mock_veo_client, sample_image_bytes, sample_video_bytes
    ):
        """Should handle multiple reference images."""
        # Create different colored images
        images = []
        for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
            img = Image.new("RGB", (100, 100), color=color)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            images.append(buffer.getvalue())

        result = generate_veo_r2v(
            client=mock_veo_client,
            reference_images=images,
            prompt="Test prompt",
        )

        assert result == sample_video_bytes
        call_kwargs = mock_veo_client.models.generate_videos.call_args
        config = call_kwargs.kwargs["config"]
        assert len(config.reference_images) == 3

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_respects_duration(self, mock_sleep, mock_veo_client, sample_image_bytes):
        """Should pass duration to config."""
        generate_veo_r2v(
            client=mock_veo_client,
            reference_images=[sample_image_bytes],
            prompt="Test prompt",
            duration=6,
        )

        call_kwargs = mock_veo_client.models.generate_videos.call_args
        config = call_kwargs.kwargs["config"]
        assert config.duration_seconds == 6

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_raises_on_empty_result(
        self, mock_sleep, mock_veo_client_empty, sample_image_bytes
    ):
        """Should raise VeoEmptyResultError when no video generated."""
        with patch.object(
            generate_veo_r2v, "__wrapped__", generate_veo_r2v.__wrapped__
        ):
            with pytest.raises(VeoEmptyResultError, match="empty result"):
                generate_veo_r2v.__wrapped__(
                    client=mock_veo_client_empty,
                    reference_images=[sample_image_bytes],
                    prompt="Test prompt",
                )

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_waits_for_completion(
        self,
        mock_sleep,
        mock_veo_client_pending,
        sample_image_bytes,
        sample_video_bytes,
    ):
        """Should poll until operation completes."""
        result = generate_veo_r2v(
            client=mock_veo_client_pending,
            reference_images=[sample_image_bytes],
            prompt="Test prompt",
        )

        assert result == sample_video_bytes
        mock_veo_client_pending.operations.get.assert_called()

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_uses_correct_model(self, mock_sleep, mock_veo_client, sample_image_bytes):
        """Should use specified model."""
        generate_veo_r2v(
            client=mock_veo_client,
            reference_images=[sample_image_bytes],
            prompt="Test prompt",
            model="veo-custom-model",
        )

        call_kwargs = mock_veo_client.models.generate_videos.call_args
        assert call_kwargs.kwargs["model"] == "veo-custom-model"

    @patch("workflows.shared.veo_utils.time.sleep")
    def test_generate_audio_disabled_by_default(
        self, mock_sleep, mock_veo_client, sample_image_bytes
    ):
        """Should have audio disabled by default."""
        generate_veo_r2v(
            client=mock_veo_client,
            reference_images=[sample_image_bytes],
            prompt="Test prompt",
        )

        call_kwargs = mock_veo_client.models.generate_videos.call_args
        config = call_kwargs.kwargs["config"]
        assert not config.generate_audio
