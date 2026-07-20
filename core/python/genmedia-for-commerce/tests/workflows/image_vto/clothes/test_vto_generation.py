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

"""Tests for vto_generation.py - VTO generation, preprocessing, and evaluation with mocks."""

from unittest.mock import Mock, patch

import pytest
from workflows.image_vto.clothes.vto_generation import (
    _build_description_block,
    evaluate_vto_image,
    generate_vto,
    preprocess_face_image,
    preprocess_model_image,
)


@pytest.fixture
def mock_client():
    """Create a mock Gemini client."""
    return Mock()


@pytest.fixture
def img_bytes():
    """Minimal image bytes."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


# ---------------------------------------------------------------------------
# _build_description_block (pure logic, no mocks needed)
# ---------------------------------------------------------------------------
class TestBuildDescriptionBlock:
    """Tests for _build_description_block helper."""

    def test_returns_empty_for_none_or_empty(self):
        """Should return empty string when descriptions is None, empty, or all-empty."""
        assert _build_description_block(None) == ""
        assert _build_description_block([]) == ""
        assert _build_description_block([{"general": "", "details": ""}]) == ""

    def test_single_garment_uses_garment_label(self):
        """Single-garment list should use 'GARMENT' (no number) and include general text."""
        result = _build_description_block(
            [{"general": "Nike running shorts", "details": ""}]
        )
        assert "**GARMENT:**" in result
        assert "GARMENT 1" not in result
        assert "Nike running shorts" in result

    def test_multiple_garments_use_numbered_labels(self):
        """Multiple garments should use 'GARMENT 1', 'GARMENT 2', etc."""
        descs = [
            {"general": "Shirt", "details": ""},
            {"general": "Pants", "details": ""},
        ]
        result = _build_description_block(descs)
        assert "**GARMENT 1:**" in result
        assert "**GARMENT 2:**" in result

    def test_filters_only_exterior_details(self):
        """Should include EXTERIOR lines but not INTERIOR lines, with IMPORTANT notice."""
        details = (
            "[EXTERIOR] White swoosh on left thigh\n"
            "[INTERIOR] Size tag at waistband\n"
            "[EXTERIOR] Brand text on back"
        )
        result = _build_description_block(
            [{"general": "Shorts", "details": details}]
        )
        assert "[EXTERIOR] White swoosh" in result
        assert "[EXTERIOR] Brand text" in result
        assert "[INTERIOR] Size tag" not in result
        assert "IMPORTANT" in result


# ---------------------------------------------------------------------------
# preprocess_face_image
# ---------------------------------------------------------------------------
class TestPreprocessFaceImage:
    """Tests for preprocess_face_image function."""

    @patch("workflows.image_vto.clothes.vto_generation.save_debug_image")
    @patch("workflows.image_vto.clothes.vto_generation.replace_background")
    @patch("workflows.image_vto.clothes.vto_generation.upscale_image_bytes")
    @patch("workflows.image_vto.clothes.vto_generation.crop_face")
    def test_returns_none_tuple_when_no_face_detected(
        self,
        mock_crop,
        mock_upscale,
        mock_replace_bg,
        mock_save,
        mock_client,
        img_bytes,
    ):
        """Should return (None, None) when crop_face finds no face."""
        mock_crop.return_value = None

        ref, preproc = preprocess_face_image(mock_client, img_bytes)

        assert ref is None
        assert preproc is None

    @patch("workflows.image_vto.clothes.vto_generation.save_debug_image")
    @patch("workflows.image_vto.clothes.vto_generation.replace_background")
    @patch("workflows.image_vto.clothes.vto_generation.upscale_image_bytes")
    @patch("workflows.image_vto.clothes.vto_generation.crop_face")
    def test_returns_reference_and_preprocessed_on_success(
        self,
        mock_crop,
        mock_upscale,
        mock_replace_bg,
        mock_save,
        mock_client,
        img_bytes,
    ):
        """Should return (reference_face, preprocessed_face) on success."""
        cropped = b"cropped_face"
        upscaled = b"upscaled_face"
        bg_removed = b"bg_removed_face"
        mock_crop.return_value = cropped
        mock_upscale.return_value = upscaled
        mock_replace_bg.return_value = bg_removed

        ref, preproc = preprocess_face_image(mock_client, img_bytes)

        assert ref == upscaled
        assert preproc == bg_removed

    @patch("workflows.image_vto.clothes.vto_generation.save_debug_image")
    @patch("workflows.image_vto.clothes.vto_generation.replace_background")
    @patch("workflows.image_vto.clothes.vto_generation.upscale_image_bytes")
    @patch("workflows.image_vto.clothes.vto_generation.crop_face")
    def test_graceful_fallbacks_on_upscale_or_bg_failure(
        self,
        mock_crop,
        mock_upscale,
        mock_replace_bg,
        mock_save,
        mock_client,
        img_bytes,
    ):
        """Should fall back gracefully when upscale or background removal fails."""
        cropped = b"cropped_face"
        mock_crop.return_value = cropped

        # Upscale fails -> reference should be cropped
        mock_upscale.side_effect = RuntimeError("Upscale API down")
        mock_replace_bg.return_value = b"bg_removed"
        ref, preproc = preprocess_face_image(mock_client, img_bytes)
        assert ref == cropped

        # Reset: upscale works, bg removal fails -> preprocessed should be reference
        mock_upscale.side_effect = None
        mock_upscale.return_value = b"upscaled"
        mock_replace_bg.side_effect = RuntimeError("BG removal failed")
        ref, preproc = preprocess_face_image(mock_client, img_bytes)
        assert ref == b"upscaled"
        assert preproc == b"upscaled"


# ---------------------------------------------------------------------------
# preprocess_model_image
# ---------------------------------------------------------------------------
class TestPreprocessModelImage:
    """Tests for preprocess_model_image function."""

    @patch("workflows.image_vto.clothes.vto_generation.save_debug_image")
    @patch("workflows.image_vto.clothes.vto_generation.upscale_image_bytes")
    @patch("workflows.image_vto.clothes.vto_generation.replace_background")
    def test_returns_upscaled_image_on_success(
        self, mock_replace_bg, mock_upscale, mock_save, mock_client, img_bytes
    ):
        """Should return the upscaled image after background removal."""
        mock_replace_bg.return_value = b"no_bg"
        mock_upscale.return_value = b"upscaled_body"

        result = preprocess_model_image(mock_client, img_bytes)

        assert result == b"upscaled_body"

    @patch("workflows.image_vto.clothes.vto_generation.save_debug_image")
    @patch("workflows.image_vto.clothes.vto_generation.upscale_image_bytes")
    @patch("workflows.image_vto.clothes.vto_generation.replace_background")
    def test_returns_original_on_error(
        self, mock_replace_bg, mock_upscale, mock_save, mock_client, img_bytes
    ):
        """Should return the original image when preprocessing fails."""
        mock_replace_bg.side_effect = RuntimeError("Error")

        result = preprocess_model_image(mock_client, img_bytes)

        assert result == img_bytes


# ---------------------------------------------------------------------------
# generate_vto
# ---------------------------------------------------------------------------
class TestGenerateVto:
    """Tests for generate_vto function."""

    @patch("workflows.image_vto.clothes.vto_generation.save_debug_image")
    @patch("workflows.image_vto.clothes.vto_generation.generate_nano")
    def test_returns_both_steps_on_success(
        self, mock_nano, mock_save, mock_client, img_bytes
    ):
        """Should return step1 and step2 images when both succeed."""
        mock_nano.side_effect = [b"step1_img", b"step2_img"]

        result = generate_vto(
            mock_client,
            scenario="studio",
            garment_images=[img_bytes],
            preprocessed_person_images=[img_bytes, img_bytes],
        )

        assert result["step1_image"] == b"step1_img"
        assert result["step2_image"] == b"step2_img"

    @patch("workflows.image_vto.clothes.vto_generation.save_debug_image")
    @patch("workflows.image_vto.clothes.vto_generation.generate_nano")
    def test_returns_none_when_step1_fails(
        self, mock_nano, mock_save, mock_client, img_bytes
    ):
        """Should return None when step 1 generation fails."""
        mock_nano.return_value = None

        result = generate_vto(
            mock_client,
            scenario="studio",
            garment_images=[img_bytes],
            preprocessed_person_images=[img_bytes, img_bytes],
        )

        assert result is None

    @patch("workflows.image_vto.clothes.vto_generation.save_debug_image")
    @patch("workflows.image_vto.clothes.vto_generation.generate_nano")
    def test_returns_step2_none_when_step2_fails(
        self, mock_nano, mock_save, mock_client, img_bytes
    ):
        """Should return step1 image with step2=None when step 2 fails."""
        mock_nano.side_effect = [b"step1_img", None]

        result = generate_vto(
            mock_client,
            scenario="studio",
            garment_images=[img_bytes],
            preprocessed_person_images=[img_bytes, img_bytes],
        )

        assert result["step1_image"] == b"step1_img"
        assert result["step2_image"] is None


# ---------------------------------------------------------------------------
# evaluate_vto_image
# ---------------------------------------------------------------------------
class TestEvaluateVtoImage:
    """Tests for evaluate_vto_image function."""

    @patch("workflows.image_vto.clothes.vto_generation.submit_evaluation")
    def test_returns_evaluation_result(self, mock_submit, img_bytes):
        """Should return the evaluation result from submit_evaluation."""
        expected = {
            "similarity_percentage": 85.0,
            "distance": 0.3,
            "model": "InsightFace-ArcFace",
            "face_detected": True,
        }
        mock_future = Mock()
        mock_future.result.return_value = expected
        mock_submit.return_value = mock_future

        result = evaluate_vto_image(img_bytes, img_bytes)

        assert result == expected

    @patch("workflows.image_vto.clothes.vto_generation.submit_evaluation")
    def test_returns_error_result_on_exception(self, mock_submit, img_bytes):
        """Should return error dict when evaluation raises."""
        mock_future = Mock()
        mock_future.result.side_effect = RuntimeError("Eval failed")
        mock_submit.return_value = mock_future

        result = evaluate_vto_image(img_bytes, img_bytes)

        assert result["similarity_percentage"] == 0.0
        assert result["distance"] == 2.0
        assert result["face_detected"] is False
        assert "error" in result
