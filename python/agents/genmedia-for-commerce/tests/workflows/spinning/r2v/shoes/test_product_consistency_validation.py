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

"""Tests for product_consistency_validation.py - SSIM and frame matching."""

import io

import numpy as np
import pytest
from PIL import Image

from workflows.spinning.r2v.shoes.product_consistency_validation import (
    bytes_to_numpy,
    calculate_ssim,
    find_best_matches_with_ssim,
    forward_fill_labels,
    match_frames_to_references,
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
def similar_red_image_bytes():
    """Create a 100x100 slightly different red image."""
    img = Image.new("RGB", (100, 100), (250, 5, 5))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestBytesToNumpy:
    """Tests for bytes_to_numpy function."""

    def test_converts_png_bytes(self, red_image_bytes):
        """Should convert PNG bytes to numpy array."""
        result = bytes_to_numpy(red_image_bytes)

        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)

    def test_bgr_format(self, red_image_bytes):
        """Result should be in BGR format (OpenCV convention)."""
        result = bytes_to_numpy(red_image_bytes)

        # Red in BGR is (0, 0, 255)
        assert result[50, 50, 2] == 255  # Red channel
        assert result[50, 50, 0] == 0  # Blue channel

    def test_invalid_bytes_raises(self):
        """Should raise ValueError for invalid bytes."""
        with pytest.raises(ValueError, match="Failed to decode"):
            bytes_to_numpy(b"not an image")


class TestCalculateSsim:
    """Tests for calculate_ssim function."""

    def test_identical_images(self, red_image_bytes):
        """Identical images should have SSIM of 1.0."""
        score = calculate_ssim(red_image_bytes, red_image_bytes)
        assert score == pytest.approx(1.0, abs=0.001)

    def test_different_images(self, red_image_bytes, blue_image_bytes):
        """Different images should have lower SSIM than identical."""
        score = calculate_ssim(red_image_bytes, blue_image_bytes)
        assert score < 1.0  # Not identical

    def test_similar_images(self, red_image_bytes, similar_red_image_bytes):
        """Similar images should have high SSIM."""
        score = calculate_ssim(red_image_bytes, similar_red_image_bytes)
        assert score > 0.9

    def test_symmetric(self, red_image_bytes, blue_image_bytes):
        """SSIM should be symmetric."""
        score1 = calculate_ssim(red_image_bytes, blue_image_bytes)
        score2 = calculate_ssim(blue_image_bytes, red_image_bytes)
        assert score1 == pytest.approx(score2, abs=0.001)

    def test_handles_different_sizes(self, red_image_bytes):
        """Should handle images of different sizes by resizing."""
        # Create a larger image
        img = Image.new("RGB", (200, 200), (255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        large_red = buffer.getvalue()

        # Should not raise, and similar images should have high score
        score = calculate_ssim(red_image_bytes, large_red)
        assert score > 0.8


class TestForwardFillLabels:
    """Tests for forward_fill_labels function."""

    def test_no_nulls(self):
        """Labels without nulls should be unchanged."""
        labels = ["left", "front", "right"]
        indices = [0, 10, 20]

        filled_labels, filled_indices = forward_fill_labels(labels, indices)

        assert filled_labels == labels
        assert filled_indices == indices

    def test_fills_nulls(self):
        """Should forward-fill null labels."""
        labels = ["left", None, None, "right", None]
        indices = [0, 6, 12, 18, 24]

        filled_labels, filled_indices = forward_fill_labels(labels, indices)

        assert filled_labels == ["left", "left", "left", "right", "right"]
        assert filled_indices == [0, 6, 12, 18, 24]

    def test_skips_leading_nulls(self):
        """Should skip nulls before first valid label."""
        labels = [None, None, "front", None]
        indices = [0, 5, 10, 15]

        filled_labels, filled_indices = forward_fill_labels(labels, indices)

        assert filled_labels == ["front", "front"]
        assert filled_indices == [10, 15]

    def test_empty_input(self):
        """Empty input should return empty output."""
        filled_labels, filled_indices = forward_fill_labels([], [])
        assert filled_labels == []
        assert filled_indices == []

    def test_handles_empty_strings(self):
        """Empty strings should be treated as null."""
        labels = ["left", "", "right"]
        indices = [0, 10, 20]

        filled_labels, _filled_indices = forward_fill_labels(labels, indices)

        assert filled_labels == ["left", "left", "right"]


class TestMatchFramesToReferences:
    """Tests for match_frames_to_references function."""

    def test_matches_labels(self):
        """Should match frame labels to reference labels."""
        frame_labels = ["right", "front", "left"]
        frame_indices = [0, 10, 20]
        reference_labels = ["right", "left", "front"]

        result = match_frames_to_references(
            frame_labels, frame_indices, reference_labels
        )

        assert len(result) == 3
        # Check that matches are correct
        right_match = next(m for m in result if m["frame_label"] == "right")
        assert right_match["reference_index"] == 0

    def test_skips_unmatched_labels(self):
        """Should skip labels not in references."""
        frame_labels = ["right", "unknown", "left"]
        frame_indices = [0, 10, 20]
        reference_labels = ["right", "left"]

        result = match_frames_to_references(
            frame_labels, frame_indices, reference_labels
        )

        assert len(result) == 2
        labels_matched = [m["frame_label"] for m in result]
        assert "unknown" not in labels_matched

    def test_skips_null_labels(self):
        """Should skip null/empty labels."""
        frame_labels = ["right", None, "", "left"]
        frame_indices = [0, 10, 20, 30]
        reference_labels = ["right", "left"]

        result = match_frames_to_references(
            frame_labels, frame_indices, reference_labels
        )

        assert len(result) == 2

    def test_uses_first_reference_for_duplicate_labels(self):
        """Should use first reference index for duplicate labels."""
        frame_labels = ["right"]
        frame_indices = [0]
        reference_labels = ["right", "right"]  # Duplicate

        result = match_frames_to_references(
            frame_labels, frame_indices, reference_labels
        )

        assert len(result) == 1
        assert result[0]["reference_index"] == 0  # First one


class TestFindBestMatchesWithSsim:
    """Tests for find_best_matches_with_ssim function."""

    def test_finds_best_match(
        self, red_image_bytes, blue_image_bytes, similar_red_image_bytes
    ):
        """Should find the frame most similar to reference."""
        # All frames
        all_frames = [blue_image_bytes, similar_red_image_bytes, blue_image_bytes]

        # Reference is red
        reference_frames = [red_image_bytes]

        # Matched pairs point to frames 0, 1, 2 for reference 0
        matched_pairs = [
            {"frame_index": 0, "frame_label": "right", "reference_index": 0},
            {"frame_index": 1, "frame_label": "right", "reference_index": 0},
            {"frame_index": 2, "frame_label": "right", "reference_index": 0},
        ]

        result = find_best_matches_with_ssim(
            matched_pairs, all_frames, reference_frames, offset_frames=0
        )

        # Should find frame 1 (similar_red) as best match
        best_matches = [r for r in result if r.get("is_best_match")]
        assert len(best_matches) == 1
        assert best_matches[0]["frame_index"] == 1

    def test_includes_offset_frames(self, red_image_bytes):
        """Should include frames at offset positions."""
        # 10 identical frames
        all_frames = [red_image_bytes] * 10
        reference_frames = [red_image_bytes]

        matched_pairs = [
            {"frame_index": i, "frame_label": "right", "reference_index": 0}
            for i in range(10)
        ]

        result = find_best_matches_with_ssim(
            matched_pairs, all_frames, reference_frames, offset_frames=2
        )

        # Should have best, before, and after
        offset_types = [r["offset_type"] for r in result]
        assert "best" in offset_types
        # May or may not have before/after depending on best match position
