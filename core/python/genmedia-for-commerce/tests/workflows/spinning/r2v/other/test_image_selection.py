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

"""Tests for image_selection.py - Generic product image classification and selection."""

import json
from unittest.mock import Mock, patch

import pytest
from workflows.spinning.r2v.other.image_selection import (
    classify_product_images,
    select_best_images,
)


@pytest.fixture
def dummy_images():
    """Six dummy image byte strings."""
    return [f"img_{i}".encode() for i in range(6)]


def _mock_classify_response(product_type, views):
    """Build a JSON string mimicking Gemini classify output."""
    classifications = [
        {"index": i, "view": v, "quality": 8} for i, v in enumerate(views)
    ]
    return json.dumps(
        {"product_type": product_type, "classifications": classifications}
    )


class TestClassifyProductImages:
    """Tests for classify_product_images function."""

    @patch("workflows.spinning.r2v.other.image_selection.generate_gemini")
    def test_valid_response_parsed(self, mock_gemini, dummy_images):
        """Should parse a well-formed Gemini response correctly."""
        mock_gemini.return_value = _mock_classify_response(
            "shoes", ["right", "left", "front", "back", "other", "right"]
        )
        mock_client = Mock()

        result = classify_product_images(mock_client, dummy_images)

        assert result["product_type"] == "shoes"
        assert len(result["classifications"]) == 6
        assert result["classifications"][0]["view"] == "right"

    @patch("workflows.spinning.r2v.other.image_selection.generate_gemini")
    def test_invalid_product_type_defaults_to_other(
        self, mock_gemini, dummy_images
    ):
        """Unknown product_type should fall back to 'other'."""
        mock_gemini.return_value = json.dumps(
            {
                "product_type": "spaceship",
                "classifications": [{"index": 0, "view": "front", "quality": 7}]
                * 6,
            }
        )
        mock_client = Mock()

        result = classify_product_images(mock_client, dummy_images)

        assert result["product_type"] == "other"

    @patch("workflows.spinning.r2v.other.image_selection.generate_gemini")
    def test_invalid_view_defaults_to_other(self, mock_gemini, dummy_images):
        """Unknown view labels should fall back to 'other'."""
        mock_gemini.return_value = json.dumps(
            {
                "product_type": "other",
                "classifications": [
                    {"index": 0, "view": "diagonal", "quality": 7}
                ]
                * 6,
            }
        )
        mock_client = Mock()

        result = classify_product_images(mock_client, dummy_images)

        for c in result["classifications"]:
            assert c["view"] == "other"

    @patch("workflows.spinning.r2v.other.image_selection.generate_gemini")
    def test_missing_classifications_padded(self, mock_gemini, dummy_images):
        """If Gemini returns fewer classifications than images, missing ones are padded."""
        mock_gemini.return_value = json.dumps(
            {
                "product_type": "other",
                "classifications": [
                    {"index": 0, "view": "front", "quality": 9}
                ],
            }
        )
        mock_client = Mock()

        result = classify_product_images(mock_client, dummy_images)

        assert len(result["classifications"]) == 6
        # Padded entries default to view="other", quality=5
        assert result["classifications"][5]["view"] == "other"
        assert result["classifications"][5]["quality"] == 5

    @patch("workflows.spinning.r2v.other.image_selection.generate_gemini")
    def test_exception_returns_safe_fallback(self, mock_gemini, dummy_images):
        """On exception, should return all-other fallback."""
        mock_gemini.side_effect = RuntimeError("API down")
        mock_client = Mock()

        result = classify_product_images(mock_client, dummy_images)

        assert result["product_type"] == "other"
        assert len(result["classifications"]) == 6


class TestSelectBestImages:
    """Tests for select_best_images function."""

    def test_four_or_fewer_returns_as_is(self):
        """4 or fewer images should be returned unchanged."""
        imgs = [b"a", b"b", b"c"]
        mock_client = Mock()

        result = select_best_images(mock_client, imgs)

        assert result == imgs

    @patch(
        "workflows.spinning.r2v.other.image_selection.classify_product_images"
    )
    def test_shoes_side_priority(self, mock_classify, dummy_images):
        """For shoes, slot 1 should prefer right, slot 2 should prefer left."""
        mock_classify.return_value = {
            "product_type": "shoes",
            "classifications": [
                {"index": 0, "view": "right", "quality": 9},
                {"index": 1, "view": "left", "quality": 8},
                {"index": 2, "view": "front", "quality": 7},
                {"index": 3, "view": "back", "quality": 6},
                {"index": 4, "view": "other", "quality": 5},
                {"index": 5, "view": "right", "quality": 4},
            ],
        }
        mock_client = Mock()

        result = select_best_images(mock_client, dummy_images)

        assert len(result) == 4
        # Slot 1 = best right (index 0), Slot 2 = best left (index 1)
        assert result[0] == dummy_images[0]
        assert result[1] == dummy_images[1]
        # Slot 3 = best front (index 2), Slot 4 = best back (index 3)
        assert result[2] == dummy_images[2]
        assert result[3] == dummy_images[3]

    @patch(
        "workflows.spinning.r2v.other.image_selection.classify_product_images"
    )
    def test_other_front_priority(self, mock_classify, dummy_images):
        """For 'other' products, slot 1 should prefer front, slot 2 should prefer back."""
        mock_classify.return_value = {
            "product_type": "other",
            "classifications": [
                {"index": 0, "view": "front", "quality": 9},
                {"index": 1, "view": "back", "quality": 8},
                {"index": 2, "view": "right", "quality": 7},
                {"index": 3, "view": "left", "quality": 6},
                {"index": 4, "view": "other", "quality": 5},
                {"index": 5, "view": "front", "quality": 4},
            ],
        }
        mock_client = Mock()

        result = select_best_images(mock_client, dummy_images)

        assert len(result) == 4
        assert result[0] == dummy_images[0]  # front
        assert result[1] == dummy_images[1]  # back

    @patch(
        "workflows.spinning.r2v.other.image_selection.classify_product_images"
    )
    def test_no_duplicate_indices_selected(self, mock_classify, dummy_images):
        """Each selected image should come from a unique index."""
        mock_classify.return_value = {
            "product_type": "other",
            "classifications": [
                {"index": 0, "view": "front", "quality": 9},
                {"index": 1, "view": "front", "quality": 8},
                {"index": 2, "view": "front", "quality": 7},
                {"index": 3, "view": "front", "quality": 6},
                {"index": 4, "view": "front", "quality": 5},
                {"index": 5, "view": "front", "quality": 4},
            ],
        }
        mock_client = Mock()

        result = select_best_images(mock_client, dummy_images)

        # All selected images should be distinct
        assert len(result) == len({id(r) for r in result})
