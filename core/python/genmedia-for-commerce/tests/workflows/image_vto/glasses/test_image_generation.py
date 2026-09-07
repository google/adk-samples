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

"""Tests for image_generation.py - Preprocess functions and describe_glasses."""

from unittest.mock import Mock, patch

import pytest
from workflows.image_vto.glasses.image_generation import (
    VTO_BACKGROUND_COLOR,
    describe_glasses,
    preprocess_face_image,
    preprocess_glasses_image,
)


@pytest.fixture
def mock_client():
    """Create a mock Gemini client."""
    return Mock()


@pytest.fixture
def fake_image_bytes():
    """Minimal fake image bytes for test arguments."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


class TestPreprocessFaceImage:
    """Tests for preprocess_face_image."""

    @patch("workflows.image_vto.glasses.image_generation.replace_background")
    @patch("workflows.image_vto.glasses.image_generation.upscale_image_bytes")
    @patch("workflows.image_vto.glasses.image_generation.crop_face")
    def test_returns_reference_and_preprocessed(
        self,
        mock_crop,
        mock_upscale,
        mock_replace_bg,
        mock_client,
        fake_image_bytes,
    ):
        """Should return (reference_face, preprocessed_face) on success."""
        cropped = b"cropped_face"
        upscaled = b"upscaled_face"
        bg_removed = b"bg_removed_face"

        mock_crop.return_value = cropped
        mock_upscale.return_value = upscaled
        mock_replace_bg.return_value = bg_removed

        ref, preprocessed = preprocess_face_image(mock_client, fake_image_bytes)

        assert ref == upscaled
        assert preprocessed == bg_removed
        mock_crop.assert_called_once_with(fake_image_bytes)
        mock_upscale.assert_called_once_with(
            mock_client, cropped, upscale_factor="x4"
        )
        mock_replace_bg.assert_called_once_with(
            mock_client,
            upscaled,
            0.01,
            VTO_BACKGROUND_COLOR,
            mask_margin_pixels=2,
            feather_radius=3,
        )

    @patch("workflows.image_vto.glasses.image_generation.crop_face")
    def test_returns_none_tuple_when_no_face_detected(
        self, mock_crop, mock_client, fake_image_bytes
    ):
        """Should return (None, None) when crop_face finds no face."""
        mock_crop.return_value = None

        ref, preprocessed = preprocess_face_image(mock_client, fake_image_bytes)

        assert ref is None
        assert preprocessed is None

    @patch("workflows.image_vto.glasses.image_generation.crop_face")
    def test_returns_none_tuple_on_exception(
        self, mock_crop, mock_client, fake_image_bytes
    ):
        """Should return (None, None) when an exception occurs."""
        mock_crop.side_effect = RuntimeError("segmentation fault")

        ref, preprocessed = preprocess_face_image(mock_client, fake_image_bytes)

        assert ref is None
        assert preprocessed is None


class TestPreprocessGlassesImage:
    """Tests for preprocess_glasses_image."""

    @patch("workflows.image_vto.glasses.image_generation.replace_background")
    def test_returns_bg_removed_image(
        self, mock_replace_bg, mock_client, fake_image_bytes
    ):
        """Should return image with background removed."""
        result_bytes = b"glasses_no_bg"
        mock_replace_bg.return_value = result_bytes

        result = preprocess_glasses_image(mock_client, fake_image_bytes)

        assert result == result_bytes
        mock_replace_bg.assert_called_once_with(
            mock_client, fake_image_bytes, 0.01, None
        )

    @patch("workflows.image_vto.glasses.image_generation.replace_background")
    def test_returns_original_on_exception(
        self, mock_replace_bg, mock_client, fake_image_bytes
    ):
        """Should fall back to original image bytes on error."""
        mock_replace_bg.side_effect = RuntimeError("background removal failed")

        result = preprocess_glasses_image(mock_client, fake_image_bytes)

        assert result == fake_image_bytes


class TestDescribeGlasses:
    """Tests for describe_glasses."""

    @patch(
        "workflows.image_vto.glasses.image_generation.get_generate_content_config"
    )
    @patch("workflows.image_vto.glasses.image_generation.generate_gemini")
    def test_returns_description_string(
        self, mock_generate, mock_config, mock_client, fake_image_bytes
    ):
        """Should return the description string from Gemini."""
        mock_config.return_value = Mock()
        mock_generate.return_value = "Ray-Ban Aviator with gold metal frame and dark green G-15 tinted lenses"

        result = describe_glasses(mock_client, fake_image_bytes)

        assert (
            result
            == "Ray-Ban Aviator with gold metal frame and dark green G-15 tinted lenses"
        )
        mock_generate.assert_called_once()

    @patch(
        "workflows.image_vto.glasses.image_generation.get_generate_content_config"
    )
    @patch("workflows.image_vto.glasses.image_generation.generate_gemini")
    def test_returns_none_on_exception(
        self, mock_generate, mock_config, mock_client, fake_image_bytes
    ):
        """Should return None when Gemini call fails."""
        mock_config.return_value = Mock()
        mock_generate.side_effect = RuntimeError("model overloaded")

        result = describe_glasses(mock_client, fake_image_bytes)

        assert result is None

    @patch(
        "workflows.image_vto.glasses.image_generation.get_generate_content_config"
    )
    @patch("workflows.image_vto.glasses.image_generation.generate_gemini")
    def test_uses_zero_temperature(
        self, mock_generate, mock_config, mock_client, fake_image_bytes
    ):
        """Should request temperature=0 for deterministic description."""
        mock_config.return_value = Mock()
        mock_generate.return_value = "Some glasses"

        describe_glasses(mock_client, fake_image_bytes)

        mock_config.assert_called_once()
        _, kwargs = mock_config.call_args
        assert kwargs["temperature"] == 0
