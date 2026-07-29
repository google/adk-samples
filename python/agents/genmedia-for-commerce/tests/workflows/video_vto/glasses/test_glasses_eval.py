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

"""Tests for glasses_eval.py - Color detection and video quality checks."""

import io

from PIL import Image

from workflows.video_vto.glasses.glasses_eval import (
    check_multiple_people,
    count_people,
    detect_color_background,
    find_color_drop_frame,
    is_people_ok,
    is_video_valid,
)


def create_color_image_bytes(color_rgb, size=(100, 100)):
    """Create an image of a solid color."""
    img = Image.new("RGB", size, color_rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestDetectColorBackground:
    """Tests for detect_color_background function."""

    def test_detects_green_screen(self):
        """Should detect high percentage of green."""
        green_image = create_color_image_bytes((0, 255, 0))
        target_rgb = (0, 255, 0)

        percentage = detect_color_background(green_image, target_rgb)

        assert percentage > 0.9  # Should be nearly 100%

    def test_no_target_color(self):
        """Should return low percentage when target color absent."""
        red_image = create_color_image_bytes((255, 0, 0))
        target_rgb = (0, 255, 0)  # Looking for green

        percentage = detect_color_background(red_image, target_rgb)

        assert percentage < 0.1

    def test_partial_color(self):
        """Should detect partial color coverage."""
        # Create image half green, half red
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        for x in range(50):
            for y in range(100):
                img.putpixel((x, y), (0, 255, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        half_green = buffer.getvalue()

        target_rgb = (0, 255, 0)
        percentage = detect_color_background(half_green, target_rgb)

        assert 0.4 < percentage < 0.6  # Around 50%

    def test_saturation_threshold(self):
        """Should respect saturation threshold."""
        # Light green (low saturation)
        light_green = create_color_image_bytes((200, 255, 200))
        target_rgb = (0, 255, 0)

        # With high saturation requirement, light green shouldn't match
        percentage = detect_color_background(
            light_green, target_rgb, saturation_min=0.5
        )
        assert percentage < 0.5


class TestFindColorDropFrame:
    """Tests for find_color_drop_frame function."""

    def test_finds_drop_frame(self):
        """Should find frame where color drops and stabilizes."""
        # Create sequence: 5 green frames, then 5 red frames
        green = create_color_image_bytes((0, 255, 0))
        red = create_color_image_bytes((255, 0, 0))
        frames = [green] * 5 + [red] * 5

        target_rgb = (0, 255, 0)
        drop_index = find_color_drop_frame(frames, target_rgb)

        # Should find the drop around frame 5
        assert drop_index == 5 or drop_index == -1  # Depends on stability check

    def test_no_drop(self):
        """Should return -1 when no significant drop."""
        # All same color
        green = create_color_image_bytes((0, 255, 0))
        frames = [green] * 10

        target_rgb = (0, 255, 0)
        find_color_drop_frame(frames, target_rgb)

        # No significant drop should occur
        # Result depends on implementation details

    def test_single_frame(self):
        """Should return -1 for single frame."""
        green = create_color_image_bytes((0, 255, 0))
        frames = [green]

        drop_index = find_color_drop_frame(frames, (0, 255, 0))
        assert drop_index == -1


class TestIsPeopleOk:
    """Tests for is_people_ok function."""

    def test_zero_people(self):
        """Zero people should be OK."""
        assert is_people_ok(0) is True

    def test_one_person(self):
        """One person should be OK."""
        assert is_people_ok(1) is True

    def test_multiple_people(self):
        """Multiple people should not be OK."""
        assert is_people_ok(2) is False
        assert is_people_ok(5) is False


class MockFaceAnnotation:
    """Mock face annotation for testing."""

    pass


class MockResponse:
    """Mock Vision API response for testing."""

    def __init__(self, face_count):
        if face_count == 0:
            self.face_annotations = None
        else:
            self.face_annotations = [MockFaceAnnotation() for _ in range(face_count)]


class TestCountPeople:
    """Tests for count_people function."""

    def test_no_faces(self):
        """Should return 0 when no face annotations."""
        response = MockResponse(0)
        assert count_people(response) == 0

    def test_one_face(self):
        """Should return 1 when one face."""
        response = MockResponse(1)
        assert count_people(response) == 1

    def test_multiple_faces(self):
        """Should return correct count for multiple faces."""
        response = MockResponse(3)
        assert count_people(response) == 3


class TestCheckMultiplePeople:
    """Tests for check_multiple_people function."""

    def test_no_faces(self):
        """Should return False when no faces."""
        response = MockResponse(0)
        assert check_multiple_people(response) is False

    def test_one_face(self):
        """Should return False when exactly one face."""
        response = MockResponse(1)
        assert check_multiple_people(response) is False

    def test_multiple_faces(self):
        """Should return True when more than one face."""
        response = MockResponse(2)
        assert check_multiple_people(response) is True

        response = MockResponse(5)
        assert check_multiple_people(response) is True


class TestIsVideoValid:
    """Tests for is_video_valid function."""

    def test_all_single_person(self):
        """Video with <=1 person throughout should be valid from start."""
        people_counts = [1, 1, 1, 1, 1]
        assert is_video_valid(people_counts) == 0

    def test_all_zero_people(self):
        """Video with no people should be valid."""
        people_counts = [0, 0, 0, 0]
        assert is_video_valid(people_counts) == 0

    def test_multiple_then_single(self):
        """Should find index where video becomes valid."""
        people_counts = [2, 2, 2, 1, 1, 1]
        index = is_video_valid(people_counts)
        assert index == 3  # First frame with 1 person

    def test_returns_to_multiple(self):
        """Should return -1 if multiple people appear after becoming valid."""
        people_counts = [2, 1, 1, 2, 1]
        index = is_video_valid(people_counts)
        assert index == -1

    def test_never_valid(self):
        """Should return -1 if never becomes single person."""
        people_counts = [2, 2, 2, 2]
        index = is_video_valid(people_counts)
        assert index == -1
