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

"""Tests for images_utils.py - Frame processing and background detection."""

import io

import numpy as np
from PIL import Image

from workflows.spinning.r2v.shoes.images_utils import (
    detect_background_color,
    detect_product_bounds_from_frames,
    image_closure_selection,
    process_frames_to_target_size,
)


def create_frame_with_border(width, height, border_color, center_color, border_size=10):
    """Create a frame with colored border and different center."""
    frame = np.full((height, width, 3), border_color, dtype=np.uint8)
    frame[border_size:-border_size, border_size:-border_size] = center_color
    return frame


class TestDetectBackgroundColor:
    """Tests for detect_background_color function."""

    def test_uniform_background(self):
        """Should detect uniform background color."""
        # Create a white frame
        frame = np.full((100, 100, 3), (255, 255, 255), dtype=np.uint8)

        result = detect_background_color(frame)

        assert result == (255, 255, 255)

    def test_frame_with_product_in_center(self):
        """Should detect border color, not center product."""
        # White border, red center (product)
        frame = create_frame_with_border(100, 100, (255, 255, 255), (0, 0, 255))

        result = detect_background_color(frame)

        # Should detect the white border, not the red center
        assert result[0] > 200  # B channel
        assert result[1] > 200  # G channel
        assert result[2] > 200  # R channel

    def test_colored_background(self):
        """Should detect colored background."""
        # Gray background
        frame = np.full((100, 100, 3), (128, 128, 128), dtype=np.uint8)

        result = detect_background_color(frame)

        assert result == (128, 128, 128)


class TestDetectProductBoundsFromFrames:
    """Tests for detect_product_bounds_from_frames function."""

    def test_empty_frames(self):
        """Should return None for empty frames list."""
        result = detect_product_bounds_from_frames([])
        assert result is None

    def test_detects_product_in_center(self):
        """Should detect product bounds in center of frame."""
        # Create frames with a product (black square) in center
        frames = []
        for _ in range(5):
            frame = np.full((200, 200, 3), (255, 255, 255), dtype=np.uint8)
            # Add a black product in center
            frame[50:150, 50:150] = (0, 0, 0)
            frames.append(frame)

        result = detect_product_bounds_from_frames(frames, sample_frames=5)

        assert result is not None
        assert "x" in result
        assert "y" in result
        assert "width" in result
        assert "height" in result
        # Product should be roughly 100x100 plus margins
        assert result["width"] >= 90
        assert result["height"] >= 90

    def test_samples_subset_of_frames(self):
        """Should sample subset when more frames than sample_frames."""
        frames = [np.full((100, 100, 3), 255, dtype=np.uint8) for _ in range(100)]
        # Add product to all frames
        for frame in frames:
            frame[30:70, 30:70] = 0

        # Should work with sampling
        result = detect_product_bounds_from_frames(frames, sample_frames=10)
        assert result is not None


class TestImageClosureSelection:
    """Tests for image_closure_selection function."""

    def test_selects_desirable_classifications(self):
        """Should select images with desirable classifications."""
        bytes_list = [b"img1", b"img2", b"img3", b"img4", b"img5"]
        classes = ["top_front", "front", "right", "left", "back"]

        result = image_closure_selection(bytes_list, classes)

        # Should select front, right, left (in order of desirability)
        assert len(result) <= 4
        assert b"img2" in result  # front
        assert b"img3" in result  # right

    def test_limits_to_four(self):
        """Should return at most 4 images."""
        bytes_list = [b"img1", b"img2", b"img3", b"img4", b"img5", b"img6"]
        classes = ["front", "right", "left", "back_left", "back_right", "back"]

        result = image_closure_selection(bytes_list, classes)

        assert len(result) <= 4

    def test_skips_unavailable_classes(self):
        """Should skip classes that aren't in desirable list."""
        bytes_list = [b"img1", b"img2"]
        classes = ["front", "right"]  # Both are in desirable list

        result = image_closure_selection(bytes_list, classes)

        assert b"img1" in result  # front
        assert b"img2" in result  # right

    def test_empty_input(self):
        """Should return empty for empty input."""
        result = image_closure_selection([], [])
        assert result == []

    def test_priority_order(self):
        """Should follow priority order: top_front, front, front_right, etc."""
        bytes_list = [b"left", b"front", b"right"]
        classes = ["left", "front", "right"]

        result = image_closure_selection(bytes_list, classes)

        # front should come before left in result (higher priority)
        # Desirable order: top_front, front, front_right, front_left, left, right, back_left, back_right
        assert b"front" in result
        if b"left" in result:
            assert result.index(b"front") < result.index(b"left")


def create_frame_bytes_with_product(
    width, height, product_rect, bg_color=(255, 255, 255), product_color=(0, 0, 0)
):
    """Create a frame with a product (colored rectangle) on a background.

    Args:
        width, height: Frame dimensions
        product_rect: (x, y, w, h) of the product rectangle
        bg_color: Background color (RGB)
        product_color: Product color (RGB)

    Returns:
        bytes: PNG encoded frame

    """
    import cv2

    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
    x, y, w, h = product_rect
    frame[y : y + h, x : x + w] = product_color
    _, encoded = cv2.imencode(".png", frame)
    return encoded.tobytes()


class TestProcessFramesToTargetSize:
    """Tests for process_frames_to_target_size function."""

    def test_processes_frames_to_target_size(self):
        """Should resize frames to target dimensions."""
        # Create frames with a product in center
        frames = []
        for _ in range(3):
            frame_bytes = create_frame_bytes_with_product(
                200,
                200,
                product_rect=(50, 50, 100, 100),
                bg_color=(255, 255, 255),
                product_color=(0, 0, 0),
            )
            frames.append(frame_bytes)

        result = process_frames_to_target_size(frames, target_size=(100, 100))

        assert len(result) == 3
        # Each frame should be PNG bytes of correct size
        for frame_bytes in result:
            assert isinstance(frame_bytes, bytes)
            img = Image.open(io.BytesIO(frame_bytes))
            assert img.size == (100, 100)

    def test_handles_tuple_target_size(self):
        """Should accept target_size as tuple."""
        frames = [create_frame_bytes_with_product(200, 200, (50, 50, 100, 100))]

        result = process_frames_to_target_size(frames, target_size=(150, 100))

        assert len(result) == 1
        img = Image.open(io.BytesIO(result[0]))
        assert img.size == (150, 100)

    def test_handles_int_target_size(self):
        """Should accept target_size as single int (square)."""
        frames = [create_frame_bytes_with_product(200, 200, (50, 50, 100, 100))]

        result = process_frames_to_target_size(frames, target_size=150)

        assert len(result) == 1
        img = Image.open(io.BytesIO(result[0]))
        assert img.size == (150, 150)

    def test_empty_input_returns_empty(self):
        """Should return empty list for empty input."""
        result = process_frames_to_target_size([], target_size=(100, 100))
        assert result == []

    def test_preserves_background_color(self):
        """Should preserve detected background color in padding."""
        # Create frame with gray background
        frame_bytes = create_frame_bytes_with_product(
            200,
            200,
            product_rect=(75, 75, 50, 50),  # Small product in center
            bg_color=(128, 128, 128),  # Gray background
            product_color=(0, 0, 255),  # Red product
        )

        result = process_frames_to_target_size([frame_bytes], target_size=(300, 300))

        assert len(result) == 1
        img = Image.open(io.BytesIO(result[0]))
        assert img.size == (300, 300)
        # Corner pixel should be close to gray (padding color)
        corner = img.getpixel((0, 0))
        # Note: OpenCV uses BGR, PIL uses RGB, so check approximately
        assert all(120 <= c <= 140 for c in corner[:3])

    def test_multiple_frames_same_size(self):
        """All output frames should have same dimensions."""
        frames = []
        for i in range(5):
            frame_bytes = create_frame_bytes_with_product(
                200,
                200,
                product_rect=(
                    40 + i * 5,
                    40 + i * 5,
                    100,
                    100,
                ),  # Slightly different positions
                bg_color=(255, 255, 255),
                product_color=(0, 0, 0),
            )
            frames.append(frame_bytes)

        result = process_frames_to_target_size(frames, target_size=(150, 150))

        assert len(result) == 5
        for frame_bytes in result:
            img = Image.open(io.BytesIO(frame_bytes))
            assert img.size == (150, 150)
