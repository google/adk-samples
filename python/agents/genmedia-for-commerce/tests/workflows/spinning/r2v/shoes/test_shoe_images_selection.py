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

"""Tests for shoe_images_selection.py - Image selection and classification."""

import io

import pytest
from PIL import Image

from workflows.spinning.r2v.shoes.shoe_images_selection import (
    can_generate_views,
    classify_video_gen_status,
    filer_top_four_views,
    get_non_white_pixel_count,
    pick_images_by_ordered_best_side,
)


@pytest.fixture
def all_white_image_bytes():
    """Create an all-white image."""
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def half_red_image_bytes():
    """Create an image that is half red, half white."""
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    # Fill left half with red
    for x in range(50):
        for y in range(100):
            img.putpixel((x, y), (255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def all_red_image_bytes():
    """Create an all-red image (no white pixels)."""
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestGetNonWhitePixelCount:
    """Tests for get_non_white_pixel_count function."""

    def test_all_white_image(self, all_white_image_bytes):
        """All white image should return 0."""
        count = get_non_white_pixel_count(all_white_image_bytes)
        assert count == 0

    def test_all_colored_image(self, all_red_image_bytes):
        """All colored image should return total pixels."""
        count = get_non_white_pixel_count(all_red_image_bytes)
        assert count == 100 * 100  # 10000 pixels

    def test_half_colored_image(self, half_red_image_bytes):
        """Half colored image should return approximately half."""
        count = get_non_white_pixel_count(half_red_image_bytes)
        assert count == 50 * 100  # 5000 pixels

    def test_invalid_bytes(self):
        """Invalid image bytes should return 0."""
        count = get_non_white_pixel_count(b"not an image")
        assert count == 0

    def test_rgba_image(self):
        """RGBA image should be handled correctly."""
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        count = get_non_white_pixel_count(buffer.getvalue())
        assert count == 10000


class TestCanGenerateViews:
    """Tests for can_generate_views function."""

    def test_all_views_present(self):
        """Should return True when all 4 views are present."""
        labels = ["front", "back", "left", "right"]
        assert can_generate_views(labels) is True

    def test_views_via_compound_labels(self):
        """Should detect views from compound labels like front_left."""
        labels = ["front_left", "back_right", "left", "right"]
        assert can_generate_views(labels) is True

    def test_missing_front(self):
        """Should return False when front is missing."""
        labels = ["back", "left", "right"]
        assert can_generate_views(labels) is False

    def test_missing_back(self):
        """Should return False when back is missing."""
        labels = ["front", "left", "right"]
        assert can_generate_views(labels) is False

    def test_missing_left(self):
        """Should return False when left is missing."""
        labels = ["front", "back", "right"]
        assert can_generate_views(labels) is False

    def test_missing_right(self):
        """Should return False when right is missing."""
        labels = ["front", "back", "left"]
        assert can_generate_views(labels) is False

    def test_top_front_not_counted_as_front(self):
        """top_front should not count as front."""
        labels = ["top_front", "back", "left", "right"]
        assert can_generate_views(labels) is False

    def test_empty_labels(self):
        """Empty labels should return False."""
        assert can_generate_views([]) is False


class TestClassifyVideoGenStatus:
    """Tests for classify_video_gen_status function."""

    def test_can_generate_returns_order(self):
        """Should return ordered list when can generate."""
        labels = ["front", "back_left", "left", "right", "front_right"]
        result = classify_video_gen_status(labels)

        assert isinstance(result, list)
        # Should be ordered according to class_order
        assert result[0] == "right"

    def test_cannot_generate_returns_exclude(self):
        """Should return 'exclude' when cannot generate views."""
        labels = ["front", "back"]  # Missing left and right
        result = classify_video_gen_status(labels)
        assert result == "exclude"


class TestFilerTopFourViews:
    """Tests for filer_top_four_views function."""

    def test_selects_four_views(self):
        """Should select up to 4 views in order: right, left, front, back."""
        selected_images = [
            (b"img_right", "right"),
            (b"img_left", "left"),
            (b"img_front", "front"),
            (b"img_back", "back"),
        ]

        result = filer_top_four_views(selected_images)

        assert len(result) == 4
        assert result[0][1] == "right"
        assert result[1][1] == "left"
        assert result[2][1] == "front"
        assert result[3][1] == "back"

    def test_uses_priority_fallback(self):
        """Should use fallback views when primary not available."""
        selected_images = [
            (b"img_front_right", "front_right"),
            (b"img_back_left", "back_left"),
            (b"img_front", "front"),
            (b"img_back", "back"),
        ]

        result = filer_top_four_views(selected_images)

        # Should use front_right for right, back_left for left
        assert len(result) == 4
        assert result[0][1] == "front_right"  # Best right
        assert result[1][1] == "back_left"  # Best left

    def test_no_duplicate_views(self):
        """Should not select the same view twice."""
        selected_images = [
            (b"img_front_right", "front_right"),
            (b"img_front", "front"),
            (b"img_back", "back"),
        ]

        result = filer_top_four_views(selected_images)

        # front_right used for right, so front should be used for front
        views_used = [r[1] for r in result]
        assert len(views_used) == len(set(views_used))


class TestPickImagesByOrderedBestSide:
    """Tests for pick_images_by_ordered_best_side function."""

    def test_picks_one_per_side(self, all_red_image_bytes, half_red_image_bytes):
        """Should pick one image per side."""
        images_classified = [
            (all_red_image_bytes, "right"),
            (half_red_image_bytes, "left"),
            (all_red_image_bytes, "front"),
            (half_red_image_bytes, "back"),
        ]

        result = pick_images_by_ordered_best_side(images_classified)

        assert len(result) == 4
        sides = [r[1] for r in result]
        assert "right" in sides
        assert "left" in sides
        assert "front" in sides
        assert "back" in sides

    def test_picks_most_complete_when_duplicates(
        self, all_red_image_bytes, half_red_image_bytes
    ):
        """Should pick image with most non-white pixels when duplicates exist."""
        images_classified = [
            (half_red_image_bytes, "right"),  # 5000 non-white
            (all_red_image_bytes, "right"),  # 10000 non-white
            (all_red_image_bytes, "left"),
            (all_red_image_bytes, "front"),
            (all_red_image_bytes, "back"),
        ]

        result = pick_images_by_ordered_best_side(images_classified)

        # Find the right image in result
        right_img = next(r for r in result if r[1] == "right")
        # Should be the all_red one (more complete)
        assert right_img[0] == all_red_image_bytes
