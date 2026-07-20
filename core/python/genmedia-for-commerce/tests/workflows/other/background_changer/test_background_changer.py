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

"""Tests for background_changer.py - Background change generation and evaluation."""

import io
from concurrent.futures import Future
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from workflows.other.background_changer.background_changer import (
    evaluate_background_change_image,
    generate_background_change,
    preprocess_face_image,
    preprocess_person_image,
)


@pytest.fixture
def face_image_bytes():
    """Create a 200x200 image simulating a face photo."""
    img = Image.new("RGB", (200, 200), color=(255, 200, 180))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def person_image_bytes():
    """Create a 400x600 image simulating a person photo."""
    img = Image.new("RGB", (400, 600), color=(100, 120, 140))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def background_image_bytes():
    """Create a 800x600 background image."""
    img = Image.new("RGB", (800, 600), color=(50, 100, 50))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestPreprocessFaceImage:
    """Tests for preprocess_face_image function."""

    @patch(
        "workflows.other.background_changer.background_changer.save_debug_image"
    )
    @patch(
        "workflows.other.background_changer.background_changer.replace_background"
    )
    @patch(
        "workflows.other.background_changer.background_changer.upscale_image_bytes"
    )
    @patch("workflows.other.background_changer.background_changer.crop_face")
    def test_returns_reference_and_preprocessed(
        self,
        mock_crop,
        mock_upscale,
        mock_replace_bg,
        mock_save,
        face_image_bytes,
    ):
        """Should return (reference_face, preprocessed_face) when face detected."""
        mock_crop.return_value = b"cropped_face"
        mock_upscale.return_value = b"upscaled_face"
        mock_replace_bg.return_value = b"bg_removed_face"

        ref, preprocessed = preprocess_face_image(
            Mock(), Mock(), face_image_bytes
        )

        assert ref == b"upscaled_face"
        assert preprocessed == b"bg_removed_face"

    @patch(
        "workflows.other.background_changer.background_changer.save_debug_image"
    )
    @patch("workflows.other.background_changer.background_changer.crop_face")
    def test_returns_none_when_no_face(
        self, mock_crop, mock_save, face_image_bytes
    ):
        """Should return (None, None) when no face is detected."""
        mock_crop.return_value = None

        ref, preprocessed = preprocess_face_image(
            Mock(), Mock(), face_image_bytes
        )

        assert ref is None
        assert preprocessed is None

    @patch(
        "workflows.other.background_changer.background_changer.save_debug_image"
    )
    @patch("workflows.other.background_changer.background_changer.crop_face")
    def test_returns_none_on_exception(
        self, mock_crop, mock_save, face_image_bytes
    ):
        """Should return (None, None) on any exception."""
        mock_crop.side_effect = RuntimeError("crop failed")

        ref, preprocessed = preprocess_face_image(
            Mock(), Mock(), face_image_bytes
        )

        assert ref is None
        assert preprocessed is None


class TestPreprocessPersonImage:
    """Tests for preprocess_person_image function."""

    @patch(
        "workflows.other.background_changer.background_changer.upscale_image_bytes"
    )
    @patch(
        "workflows.other.background_changer.background_changer.replace_background"
    )
    def test_removes_bg_then_upscales(
        self, mock_replace_bg, mock_upscale, person_image_bytes
    ):
        """Should remove background first, then upscale."""
        mock_replace_bg.return_value = b"no_bg"
        mock_upscale.return_value = b"upscaled"
        mock_client = Mock()
        mock_upscale_client = Mock()

        result = preprocess_person_image(
            mock_client, mock_upscale_client, person_image_bytes
        )

        assert result == b"upscaled"
        mock_replace_bg.assert_called_once()
        # Upscale should receive the bg-removed output
        mock_upscale.assert_called_once_with(
            mock_upscale_client, b"no_bg", upscale_factor="x4"
        )

    @patch(
        "workflows.other.background_changer.background_changer.upscale_image_bytes"
    )
    @patch(
        "workflows.other.background_changer.background_changer.replace_background"
    )
    def test_returns_original_on_error(
        self, mock_replace_bg, mock_upscale, person_image_bytes
    ):
        """Should return original image bytes when preprocessing fails."""
        mock_replace_bg.side_effect = RuntimeError("API error")

        result = preprocess_person_image(Mock(), Mock(), person_image_bytes)

        assert result == person_image_bytes


class TestGenerateBackgroundChange:
    """Tests for generate_background_change function."""

    @patch(
        "workflows.other.background_changer.background_changer.generate_nano"
    )
    def test_text_background_no_face(self, mock_nano, person_image_bytes):
        """With text background and no reference face, should return step1 only."""
        mock_nano.return_value = b"generated_image"

        result = generate_background_change(
            Mock(),
            person_image_bytes,
            reference_face=None,
            background_description="a sunny beach",
        )

        assert result == {
            "step1_image": b"generated_image",
            "step2_image": None,
        }
        mock_nano.assert_called_once()

    @patch(
        "workflows.other.background_changer.background_changer.generate_nano"
    )
    def test_image_background_no_face(
        self, mock_nano, person_image_bytes, background_image_bytes
    ):
        """With background image and no reference face, should return step1 only."""
        mock_nano.return_value = b"generated_image"

        result = generate_background_change(
            Mock(),
            person_image_bytes,
            reference_face=None,
            background_image=background_image_bytes,
        )

        assert result == {
            "step1_image": b"generated_image",
            "step2_image": None,
        }

    @patch(
        "workflows.other.background_changer.background_changer.generate_nano"
    )
    def test_with_reference_face_runs_two_steps(
        self, mock_nano, person_image_bytes, face_image_bytes
    ):
        """With reference face, should run step1 and step2 (face correction)."""
        mock_nano.side_effect = [b"step1_result", b"step2_result"]

        result = generate_background_change(
            Mock(),
            person_image_bytes,
            reference_face=face_image_bytes,
            background_description="a park",
        )

        assert result == {
            "step1_image": b"step1_result",
            "step2_image": b"step2_result",
        }
        assert mock_nano.call_count == 2

    @patch(
        "workflows.other.background_changer.background_changer.generate_nano"
    )
    def test_step1_failure_returns_none(self, mock_nano, person_image_bytes):
        """If step1 fails (returns None), result should be None."""
        mock_nano.return_value = None

        result = generate_background_change(
            Mock(),
            person_image_bytes,
            background_description="a park",
        )

        assert result is None

    @patch(
        "workflows.other.background_changer.background_changer.generate_nano"
    )
    def test_step2_failure_returns_step1_only(
        self, mock_nano, person_image_bytes, face_image_bytes
    ):
        """If step2 fails, should return step1 result with step2_image=None."""
        mock_nano.side_effect = [b"step1_result", None]

        result = generate_background_change(
            Mock(),
            person_image_bytes,
            reference_face=face_image_bytes,
            background_description="a park",
        )

        assert result == {"step1_image": b"step1_result", "step2_image": None}

    @patch(
        "workflows.other.background_changer.background_changer.generate_nano"
    )
    def test_default_background_description(
        self, mock_nano, person_image_bytes
    ):
        """When no description or image given, should use default grey studio."""
        mock_nano.return_value = b"generated"

        generate_background_change(Mock(), person_image_bytes)

        call_args = mock_nano.call_args
        pieces = call_args.kwargs["text_images_pieces"]
        # The text pieces should contain the default description
        text_content = " ".join(str(p) for p in pieces)
        assert "neutral grey studio" in text_content


class TestEvaluateBackgroundChangeImage:
    """Tests for evaluate_background_change_image function."""

    @patch(
        "workflows.other.background_changer.background_changer.submit_evaluation"
    )
    def test_returns_evaluation_result(
        self, mock_submit, face_image_bytes, person_image_bytes
    ):
        """Should return the result from the evaluation future."""
        expected = {
            "similarity_percentage": 85.0,
            "distance": 0.15,
            "model": "ArcFace",
            "face_detected": True,
        }
        future = Future()
        future.set_result(expected)
        mock_submit.return_value = future

        result = evaluate_background_change_image(
            Mock(), person_image_bytes, face_image_bytes
        )

        assert result["similarity_percentage"] == 85.0
        assert result["face_detected"] is True

    @patch(
        "workflows.other.background_changer.background_changer.submit_evaluation"
    )
    def test_returns_fallback_on_exception(
        self, mock_submit, face_image_bytes, person_image_bytes
    ):
        """Should return a safe fallback dict when evaluation raises."""
        mock_submit.side_effect = RuntimeError("pool crashed")

        result = evaluate_background_change_image(
            Mock(), person_image_bytes, face_image_bytes
        )

        assert result["similarity_percentage"] == 0.0
        assert result["face_detected"] is False
        assert "error" in result
