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

"""Tests for glasses_eval.py - Weighted scoring, clamping, and discard logic."""

import json
from unittest.mock import Mock, patch

import pytest
from workflows.image_vto.glasses.glasses_eval import (
    evaluate_all_glasses,
    evaluate_glasses,
)


@pytest.fixture
def mock_client():
    """Create a mock Gemini client."""
    return Mock()


@pytest.fixture
def fake_image_bytes():
    """Minimal fake image bytes for test arguments."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


class TestEvaluateGlassesWeightedScore:
    """Tests for the weighted average score calculation in evaluate_glasses."""

    @patch("workflows.image_vto.glasses.glasses_eval.generate_gemini")
    def test_weighted_average_all_tens(
        self, mock_generate, mock_client, fake_image_bytes
    ):
        """Perfect scores (all 10) should produce a final score of 100."""
        mock_generate.return_value = json.dumps(
            {
                "explanation": "Perfect match",
                "size": 10,
                "shape": 10,
                "color": 10,
                "lenses": 10,
            }
        )

        result = evaluate_glasses(
            mock_client, fake_image_bytes, fake_image_bytes
        )

        assert result["score"] == pytest.approx(100.0)

    @patch("workflows.image_vto.glasses.glasses_eval.generate_gemini")
    def test_weighted_average_all_zeros(
        self, mock_generate, mock_client, fake_image_bytes
    ):
        """All-zero scores should produce a final score of 0."""
        mock_generate.return_value = json.dumps(
            {
                "explanation": "Nothing matches",
                "size": 0,
                "shape": 0,
                "color": 0,
                "lenses": 0,
            }
        )

        result = evaluate_glasses(
            mock_client, fake_image_bytes, fake_image_bytes
        )

        assert result["score"] == pytest.approx(0.0)

    @patch("workflows.image_vto.glasses.glasses_eval.generate_gemini")
    def test_weighted_average_mixed_scores(
        self, mock_generate, mock_client, fake_image_bytes
    ):
        """Verify weighted average: size*0.2 + shape*0.3 + color*0.25 + lenses*0.25, times 10."""
        mock_generate.return_value = json.dumps(
            {
                "explanation": "Mixed results",
                "size": 8,
                "shape": 6,
                "color": 4,
                "lenses": 10,
            }
        )

        result = evaluate_glasses(
            mock_client, fake_image_bytes, fake_image_bytes
        )

        expected = (8 * 0.2 + 6 * 0.3 + 4 * 0.25 + 10 * 0.25) * 10
        assert result["score"] == pytest.approx(expected)


class TestEvaluateGlassesClamping:
    """Tests for score clamping to [0, 10] range."""

    @patch("workflows.image_vto.glasses.glasses_eval.generate_gemini")
    def test_clamps_scores_above_ten(
        self, mock_generate, mock_client, fake_image_bytes
    ):
        """Scores above 10 should be clamped to 10."""
        mock_generate.return_value = json.dumps(
            {
                "explanation": "Overenthusiastic LLM",
                "size": 15,
                "shape": 12,
                "color": 10,
                "lenses": 10,
            }
        )

        result = evaluate_glasses(
            mock_client, fake_image_bytes, fake_image_bytes
        )

        assert result["size"] == 10
        assert result["shape"] == 10

    @patch("workflows.image_vto.glasses.glasses_eval.generate_gemini")
    def test_clamps_negative_scores_to_zero(
        self, mock_generate, mock_client, fake_image_bytes
    ):
        """Negative scores should be clamped to 0."""
        mock_generate.return_value = json.dumps(
            {
                "explanation": "Negative scores from LLM",
                "size": -3,
                "shape": 5,
                "color": 7,
                "lenses": 8,
            }
        )

        result = evaluate_glasses(
            mock_client, fake_image_bytes, fake_image_bytes
        )

        assert result["size"] == 0


class TestEvaluateGlassesErrorHandling:
    """Tests for error handling in evaluate_glasses."""

    @patch("workflows.image_vto.glasses.glasses_eval.generate_gemini")
    def test_returns_zeros_on_api_exception(
        self, mock_generate, mock_client, fake_image_bytes
    ):
        """An API exception should return all-zero scores with error explanation."""
        mock_generate.side_effect = RuntimeError("API unavailable")

        result = evaluate_glasses(
            mock_client, fake_image_bytes, fake_image_bytes
        )

        assert result["score"] == 0.0
        assert result["size"] == 0
        assert result["shape"] == 0
        assert result["color"] == 0
        assert result["lenses"] == 0
        assert "Error" in result["explanation"]

    @patch("workflows.image_vto.glasses.glasses_eval.generate_gemini")
    def test_returns_zeros_on_missing_key(
        self, mock_generate, mock_client, fake_image_bytes
    ):
        """Missing required key in LLM response should produce error result."""
        mock_generate.return_value = json.dumps(
            {
                "explanation": "Incomplete response",
                "size": 7,
                # missing shape, color, lenses
            }
        )

        result = evaluate_glasses(
            mock_client, fake_image_bytes, fake_image_bytes
        )

        assert result["score"] == 0.0
        assert "Error" in result["explanation"]


class TestEvaluateAllGlassesDiscard:
    """Tests for discard logic in evaluate_all_glasses."""

    @patch("workflows.image_vto.glasses.glasses_eval.evaluate_glasses")
    def test_discard_when_any_attribute_is_zero(
        self, mock_eval, mock_client, fake_image_bytes
    ):
        """Should discard when any individual attribute scored 0."""
        mock_eval.return_value = {
            "explanation": "Shape is completely wrong",
            "size": 7,
            "shape": 0,
            "color": 8,
            "lenses": 6,
            "score": 42.0,
        }

        result = evaluate_all_glasses(
            mock_client, fake_image_bytes, [fake_image_bytes]
        )

        assert result["discard"] is True

    @patch("workflows.image_vto.glasses.glasses_eval.evaluate_glasses")
    def test_no_discard_when_all_attributes_nonzero(
        self, mock_eval, mock_client, fake_image_bytes
    ):
        """Should not discard when all individual attributes are above 0."""
        mock_eval.return_value = {
            "explanation": "Good match",
            "size": 7,
            "shape": 8,
            "color": 6,
            "lenses": 9,
            "score": 75.0,
        }

        result = evaluate_all_glasses(
            mock_client, fake_image_bytes, [fake_image_bytes]
        )

        assert result["discard"] is False
        assert result["glasses_score"] == 75.0

    @patch("workflows.image_vto.glasses.glasses_eval.evaluate_glasses")
    def test_uses_first_glasses_image_as_reference(
        self, mock_eval, mock_client, fake_image_bytes
    ):
        """Should pass only the first glasses image to evaluate_glasses."""
        second_image = b"second_glasses"
        mock_eval.return_value = {
            "explanation": "OK",
            "size": 5,
            "shape": 5,
            "color": 5,
            "lenses": 5,
            "score": 50.0,
        }

        evaluate_all_glasses(
            mock_client, fake_image_bytes, [fake_image_bytes, second_image]
        )

        # The third positional argument to evaluate_glasses should be the first glasses image
        call_args = mock_eval.call_args[0]
        assert call_args[2] == fake_image_bytes
