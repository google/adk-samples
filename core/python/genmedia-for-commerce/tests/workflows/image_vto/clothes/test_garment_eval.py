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

"""Tests for garment_eval.py - Garment and wearing quality evaluation with mocked Gemini."""

import json
from unittest.mock import Mock, patch

import pytest
from workflows.image_vto.clothes.garment_eval import (
    evaluate_garments,
    evaluate_wearing_quality,
)


@pytest.fixture
def mock_client():
    """Create a mock Gemini client."""
    return Mock()


@pytest.fixture
def generated_image_bytes():
    """Minimal generated image bytes."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


@pytest.fixture
def garment_image_bytes():
    """Minimal garment image bytes."""
    return b"\x89PNG\r\n\x1a\n" + b"\xff" * 20


class TestEvaluateGarments:
    """Tests for evaluate_garments function."""

    @patch("workflows.image_vto.clothes.garment_eval.evaluate_garment")
    def test_normalizes_scores_to_100_scale(
        self, mock_eval, mock_client, generated_image_bytes, garment_image_bytes
    ):
        """Should normalize 0-10 scores to 0-100 scale."""
        mock_eval.return_value = {"score": 8, "explanation": "Good match"}

        result = evaluate_garments(
            mock_client, generated_image_bytes, [garment_image_bytes]
        )

        assert result["garments_score"] == 80.0

    @patch("workflows.image_vto.clothes.garment_eval.evaluate_garment")
    def test_averages_multiple_garment_scores(
        self, mock_eval, mock_client, generated_image_bytes, garment_image_bytes
    ):
        """Should average scores across multiple garments."""
        mock_eval.side_effect = [
            {"score": 10, "explanation": "Perfect"},
            {"score": 6, "explanation": "Decent"},
        ]

        result = evaluate_garments(
            mock_client,
            generated_image_bytes,
            [garment_image_bytes, garment_image_bytes],
        )

        # Average = (10 + 6) / 2 = 8, normalized = 80
        assert result["garments_score"] == 80.0

    @patch("workflows.image_vto.clothes.garment_eval.evaluate_garment")
    def test_discard_logic_uses_threshold_of_5(
        self, mock_eval, mock_client, generated_image_bytes, garment_image_bytes
    ):
        """Should discard when any score < 5, keep when all >= 5."""
        # Score below 5 -> discard
        mock_eval.side_effect = [
            {"score": 9, "explanation": "Great"},
            {"score": 4, "explanation": "Wrong color"},
        ]
        result = evaluate_garments(
            mock_client,
            generated_image_bytes,
            [garment_image_bytes, garment_image_bytes],
        )
        assert result["discard"] is True

        # All scores >= 5 -> keep
        mock_eval.side_effect = [
            {"score": 5, "explanation": "Acceptable"},
            {"score": 7, "explanation": "Good"},
        ]
        result = evaluate_garments(
            mock_client,
            generated_image_bytes,
            [garment_image_bytes, garment_image_bytes],
        )
        assert result["discard"] is False

    @patch("workflows.image_vto.clothes.garment_eval.evaluate_garment")
    def test_returns_garment_details_list(
        self, mock_eval, mock_client, generated_image_bytes, garment_image_bytes
    ):
        """Should include individual garment evaluations in garment_details."""
        detail = {"score": 7, "explanation": "Good match"}
        mock_eval.return_value = detail

        result = evaluate_garments(
            mock_client, generated_image_bytes, [garment_image_bytes]
        )

        assert len(result["garment_details"]) == 1
        assert result["garment_details"][0] == detail

    @patch("workflows.image_vto.clothes.garment_eval.evaluate_garment")
    def test_uses_empty_descriptions_when_none_provided(
        self, mock_eval, mock_client, generated_image_bytes, garment_image_bytes
    ):
        """Should pass empty strings for view_details and garment_description when descriptions are None."""
        mock_eval.return_value = {"score": 7, "explanation": "OK"}

        evaluate_garments(
            mock_client,
            generated_image_bytes,
            [garment_image_bytes],
            garment_descriptions=None,
        )

        call_kwargs = mock_eval.call_args
        assert call_kwargs[1]["view_details"] == ""
        assert call_kwargs[1]["garment_description"] == ""

    @patch("workflows.image_vto.clothes.garment_eval.evaluate_garment")
    def test_forwards_descriptions_to_evaluate_garment(
        self, mock_eval, mock_client, generated_image_bytes, garment_image_bytes
    ):
        """Should pass garment description fields to evaluate_garment."""
        mock_eval.return_value = {"score": 8, "explanation": "Good"}

        descriptions = [
            {"general": "Blue shirt", "details": "[EXTERIOR] Logo on chest"}
        ]

        evaluate_garments(
            mock_client,
            generated_image_bytes,
            [garment_image_bytes],
            garment_descriptions=descriptions,
        )

        call_kwargs = mock_eval.call_args
        assert call_kwargs[1]["garment_description"] == "Blue shirt"
        assert call_kwargs[1]["view_details"] == "[EXTERIOR] Logo on chest"


class TestEvaluateWearingQuality:
    """Tests for evaluate_wearing_quality function."""

    @patch("workflows.image_vto.clothes.garment_eval.generate_gemini")
    def test_returns_valid_score_and_explanation(
        self, mock_generate, mock_client, generated_image_bytes
    ):
        """Should parse and return score and explanation from Gemini response."""
        mock_generate.return_value = json.dumps(
            {
                "explanation": "Outfit looks natural",
                "score": 3,
            }
        )

        result = evaluate_wearing_quality(mock_client, generated_image_bytes)

        assert result["score"] == 3
        assert result["explanation"] == "Outfit looks natural"

    @patch("workflows.image_vto.clothes.garment_eval.generate_gemini")
    def test_clamps_score_to_valid_range(
        self, mock_generate, mock_client, generated_image_bytes
    ):
        """Should clamp scores to the 0-3 range."""
        # Above upper bound
        mock_generate.return_value = json.dumps(
            {"explanation": "Perfect", "score": 10}
        )
        assert (
            evaluate_wearing_quality(mock_client, generated_image_bytes)[
                "score"
            ]
            == 3
        )

        # Below lower bound
        mock_generate.return_value = json.dumps(
            {"explanation": "Broken", "score": -5}
        )
        assert (
            evaluate_wearing_quality(mock_client, generated_image_bytes)[
                "score"
            ]
            == 0
        )

    @patch("workflows.image_vto.clothes.garment_eval.generate_gemini")
    def test_returns_error_result_on_failure(
        self, mock_generate, mock_client, generated_image_bytes
    ):
        """Should return score 0 on missing keys or API exception."""
        # Missing required keys
        mock_generate.return_value = json.dumps({"wrong_key": "value"})
        result = evaluate_wearing_quality(mock_client, generated_image_bytes)
        assert result["score"] == 0
        assert "Error" in result["explanation"]

        # API exception
        mock_generate.side_effect = RuntimeError("API error")
        result = evaluate_wearing_quality(mock_client, generated_image_bytes)
        assert result["score"] == 0
        assert "API error" in result["explanation"]
