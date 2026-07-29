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

"""Tests for split_multiple_shoes.py - Mask manipulation and image splitting."""

import numpy as np
from PIL import Image

from workflows.spinning.r2v.shoes.split_multiple_shoes import (
    _prepare_mask,
    check_contour_count,
    construct_image,
    filter_by_score_distribution,
    sort_masks_by_horizontal_position,
    subtract_masks,
)


class TestSortMasksByHorizontalPosition:
    """Tests for sort_masks_by_horizontal_position function."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        result = sort_masks_by_horizontal_position([])
        assert result == []

    def test_single_mask(self):
        """Single mask should return unchanged."""
        mask = [[True, True, False], [True, False, False]]
        mask_data = [(mask, 0.9)]
        result = sort_masks_by_horizontal_position(mask_data)
        assert len(result) == 1
        assert result[0][1] == 0.9

    def test_right_mode_sorts_rightmost_first(self):
        """Right mode should sort rightmost masks first."""
        # Left mask - True pixels on left side
        left_mask = [
            [True, True, False, False, False],
            [True, True, False, False, False],
        ]
        # Right mask - True pixels on right side
        right_mask = [
            [False, False, False, True, True],
            [False, False, False, True, True],
        ]

        mask_data = [(left_mask, 0.8), (right_mask, 0.9)]
        result = sort_masks_by_horizontal_position(mask_data, mode="right")

        # Right mask should come first
        assert result[0][1] == 0.9
        assert result[1][1] == 0.8

    def test_left_mode_sorts_leftmost_first(self):
        """Left mode should sort leftmost masks first."""
        left_mask = [
            [True, True, False, False, False],
            [True, True, False, False, False],
        ]
        right_mask = [
            [False, False, False, True, True],
            [False, False, False, True, True],
        ]

        mask_data = [(left_mask, 0.8), (right_mask, 0.9)]
        result = sort_masks_by_horizontal_position(mask_data, mode="left")

        # Left mask should come first
        assert result[0][1] == 0.8
        assert result[1][1] == 0.9

    def test_invalid_mode_defaults_to_right(self):
        """Invalid mode should default to 'right'."""
        mask = [[True, False], [False, True]]
        mask_data = [(mask, 0.5)]
        result = sort_masks_by_horizontal_position(
            mask_data, mode="invalid", verbose=True
        )
        assert len(result) == 1


class TestSubtractMasks:
    """Tests for subtract_masks function."""

    def test_no_overlap(self):
        """Non-overlapping masks should remain unchanged."""
        mask_a = np.array(
            [[True, True, False, False], [True, True, False, False]], dtype=bool
        )
        mask_b = np.array(
            [[False, False, True, True], [False, False, True, True]], dtype=bool
        )

        result = subtract_masks(mask_a, mask_b)
        np.testing.assert_array_equal(result, mask_a)

    def test_full_overlap(self):
        """Completely overlapping masks should result in empty mask."""
        mask_a = np.array([[True, True], [True, True]], dtype=bool)
        mask_b = np.array([[True, True], [True, True]], dtype=bool)

        result = subtract_masks(mask_a, mask_b)
        assert not result.any()

    def test_partial_overlap(self):
        """Partial overlap should remove only overlapping region."""
        mask_a = np.array([[True, True, True], [True, True, True]], dtype=bool)
        mask_b = np.array([[False, True, True], [False, True, True]], dtype=bool)

        result = subtract_masks(mask_a, mask_b)
        expected = np.array([[True, False, False], [True, False, False]], dtype=bool)
        np.testing.assert_array_equal(result, expected)


class TestPrepareMask:
    """Tests for _prepare_mask function."""

    def test_converts_bool_to_uint8(self):
        """Boolean mask should be converted to uint8 with 0/255 values."""
        mask = np.array([[True, False], [False, True]], dtype=bool)
        result = _prepare_mask(mask)

        assert result.dtype == np.uint8
        assert result[0, 0] == 255
        assert result[0, 1] == 0

    def test_handles_already_uint8(self):
        """Already uint8 mask should be binarized correctly."""
        mask = np.array([[255, 0], [128, 50]], dtype=np.uint8)
        result = _prepare_mask(mask)

        assert result.dtype == np.uint8
        assert result[0, 0] == 255
        assert result[0, 1] == 0
        assert result[1, 0] == 255  # 128 > 127
        assert result[1, 1] == 0  # 50 < 127


class TestCheckContourCount:
    """Tests for check_contour_count function."""

    def test_single_contour(self):
        """Single contour should return True."""
        # Create a mask with one blob
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 255

        result = check_contour_count(mask, max_allowed=1)
        assert result is True

    def test_multiple_contours_below_max(self):
        """Multiple contours below max should return True."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255  # First blob
        mask[50:60, 50:60] = 255  # Second blob

        result = check_contour_count(mask, max_allowed=3)
        assert result is True

    def test_too_many_contours(self):
        """Too many contours should return False."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:15, 10:15] = 255  # Blob 1
        mask[30:35, 30:35] = 255  # Blob 2
        mask[50:55, 50:55] = 255  # Blob 3
        mask[70:75, 70:75] = 255  # Blob 4

        result = check_contour_count(mask, max_allowed=3)
        assert result is False

    def test_empty_mask(self):
        """Empty mask should return True (0 contours <= max)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = check_contour_count(mask, max_allowed=1)
        assert result is True


class TestFilterByScoreDistribution:
    """Tests for filter_by_score_distribution function."""

    def test_filters_top_30_percent(self):
        """Should keep masks in top 30% of scores."""
        masks = [
            (np.zeros((10, 10)), 0.1),
            (np.zeros((10, 10)), 0.3),
            (np.zeros((10, 10)), 0.5),
            (np.zeros((10, 10)), 0.7),
            (np.zeros((10, 10)), 0.9),
        ]

        result, threshold = filter_by_score_distribution(masks, minimum=1)

        # Should keep masks with score >= 70th percentile
        assert all(item[1] >= threshold for item in result)

    def test_respects_minimum(self):
        """Should return at least minimum number of masks."""
        masks = [
            (np.zeros((10, 10)), 0.1),
            (np.zeros((10, 10)), 0.2),
            (np.zeros((10, 10)), 0.9),
        ]

        result, _ = filter_by_score_distribution(masks, minimum=2)

        # Should return at least 2 masks even if only 1 is in top 30%
        assert len(result) >= 2


class TestConstructImage:
    """Tests for construct_image function."""

    def test_creates_image_with_mask(self):
        """Should create image with mask applied and white background."""
        # Create a 100x100 red image
        image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))

        # Create a mask that only shows the center
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (25, 25, 75, 75))  # White center

        result = construct_image(image, mask)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        # Result should be cropped to mask bounds (50x50)
        assert result.size == (50, 50)

    def test_empty_mask_returns_small_image(self):
        """Empty mask should return a 1x1 white image."""
        image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        mask = Image.new("L", (100, 100), 0)  # All black/empty

        result = construct_image(image, mask)

        assert result.size == (1, 1)
        # Should be white
        assert result.getpixel((0, 0)) == (255, 255, 255, 255)

    def test_converts_rgb_to_rgba(self):
        """RGB image should be converted to RGBA."""
        image = Image.new("RGB", (100, 100), (255, 0, 0))
        mask = Image.new("L", (100, 100), 255)

        result = construct_image(image, mask)

        assert result.mode == "RGBA"
