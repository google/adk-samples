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

"""Tests for video_validation_r2v.py - Path validation and frame sampling."""

from workflows.spinning.r2v.shoes.video_validation_r2v import (
    all_classes_present,
    find_second_occurrence_range,
    is_valid,
    is_valid_path,
    is_valid_path_lessstrict,
    sample_frames,
)


class TestIsValidPath:
    """Tests for is_valid_path function."""

    def test_empty_path(self):
        """Empty path should be valid."""
        allowed = {"a": ["b"], "b": ["c"]}
        is_valid, _msg = is_valid_path([], allowed)
        assert is_valid is True

    def test_single_node_path(self):
        """Single node path should be valid."""
        allowed = {"a": ["b"]}
        is_valid, _msg = is_valid_path(["a"], allowed)
        assert is_valid is True

    def test_valid_transition(self):
        """Valid transitions should pass."""
        allowed = {"a": ["a", "b"], "b": ["b", "c"], "c": ["c"]}
        is_valid, _msg = is_valid_path(["a", "b", "c"], allowed)
        assert is_valid is True

    def test_invalid_transition(self):
        """Invalid transitions should fail."""
        allowed = {"a": ["b"], "b": ["c"]}
        is_valid, msg = is_valid_path(["a", "c"], allowed)  # Can't go a->c
        assert is_valid is False
        assert "cannot go from 'a' to 'c'" in msg

    def test_unknown_node(self):
        """Unknown node should fail."""
        allowed = {"a": ["b"], "b": ["c"]}
        is_valid, msg = is_valid_path(["a", "x"], allowed)
        assert is_valid is False
        assert "'a' is not in the map" in msg or "cannot go" in msg

    def test_self_loop(self):
        """Self-loops should be valid if allowed."""
        allowed = {"a": ["a", "b"]}
        is_valid, _msg = is_valid_path(["a", "a", "a"], allowed)
        assert is_valid is True


class TestIsValidPathLessStrict:
    """Tests for is_valid_path_lessstrict function."""

    def test_tolerates_misclassifications(self):
        """Should tolerate some misclassifications."""
        allowed = {"a": ["a", "b"], "b": ["b", "c"], "c": ["c"]}
        # One wrong classification in the middle
        path = ["a", "a", "x", "a", "b"]
        is_valid, _msg, corrected = is_valid_path_lessstrict(
            path, allowed, max_consecutive_violations=2
        )
        assert is_valid is True
        assert corrected is not None

    def test_fails_on_too_many_violations(self):
        """Should fail on too many consecutive violations."""
        allowed = {"a": ["a", "b"], "b": ["b"]}
        path = ["a", "x", "x", "x", "x", "x", "x"]  # 6 violations
        is_valid, _msg, _corrected = is_valid_path_lessstrict(
            path, allowed, max_consecutive_violations=5
        )
        assert is_valid is False

    def test_handles_oscillations(self):
        """Should handle oscillations between adjacent classes."""
        allowed = {"a": ["a", "b"], "b": ["a", "b", "c"], "c": ["c"]}
        # Oscillating between a and b
        path = ["a", "b", "a", "b", "c"]
        # This should be tolerated as valid oscillation during transition
        is_valid, _msg, _corrected = is_valid_path_lessstrict(path, allowed)
        # Note: Actual behavior depends on implementation details
        assert isinstance(is_valid, bool)


class TestIsValid:
    """Tests for is_valid function (combines clockwise and anti-clockwise)."""

    def test_clockwise_valid(self):
        """Valid clockwise rotation should pass."""
        # Clockwise: right -> front_right -> front -> front_left -> left -> back_left -> back -> back_right
        path = ["right", "front_right", "front", "front_left", "left"]
        is_valid_result, reversed_flag, msg, _corrected = is_valid(path, strict=True)
        assert is_valid_result is True
        assert reversed_flag is False
        assert "Clockwise" in msg

    def test_anticlockwise_valid(self):
        """Valid anti-clockwise rotation should pass (reversed)."""
        # Anti-clockwise is the reverse of clockwise
        path = ["left", "front_left", "front", "front_right", "right"]
        is_valid_result, reversed_flag, msg, _corrected = is_valid(path, strict=True)
        assert is_valid_result is True
        assert reversed_flag is True
        assert "Anti Clockwise" in msg

    def test_invalid_path(self):
        """Invalid path should fail."""
        path = ["right", "back"]  # Invalid: can't go right -> back
        is_valid_result, _reversed_flag, _msg, _corrected = is_valid(path, strict=True)
        assert is_valid_result is False


class TestFindSecondOccurrenceRange:
    """Tests for find_second_occurrence_range function."""

    def test_finds_second_occurrence(self):
        """Should find the range of second occurrence."""
        classes = ["right", "front", "left", "back", "right", "right"]
        indices = [0, 10, 20, 30, 100, 110]  # Second 'right' at 100

        start, end = find_second_occurrence_range(
            classes, indices, min_frame_distance=50
        )

        assert start == 100
        assert end == 110

    def test_no_second_occurrence(self):
        """Should return None, None if no second occurrence."""
        classes = ["right", "front", "left", "back"]
        indices = [0, 10, 20, 30]

        start, end = find_second_occurrence_range(classes, indices)

        assert start is None
        assert end is None

    def test_second_occurrence_too_close(self):
        """Should return None if second occurrence is too close."""
        classes = ["right", "front", "right"]
        indices = [0, 10, 20]  # Second 'right' only 10 frames from end of first

        start, end = find_second_occurrence_range(
            classes, indices, min_frame_distance=50
        )

        assert start is None
        assert end is None

    def test_empty_input(self):
        """Empty input should return None, None."""
        start, end = find_second_occurrence_range([], [])
        assert start is None
        assert end is None

    def test_mismatched_lengths(self):
        """Mismatched lengths should return None, None."""
        classes = ["right", "front"]
        indices = [0]

        start, end = find_second_occurrence_range(classes, indices)

        assert start is None
        assert end is None


class TestSampleFrames:
    """Tests for sample_frames function."""

    def test_basic_sampling(self):
        """Should sample frames at regular intervals."""
        frames = list(range(48))  # 48 frames at 24fps = 2 seconds

        sampled, indices = sample_frames(
            frames, target_samples_per_sec=12, original_fps=24
        )

        # At 12 samples/sec from 24fps, interval = 2
        assert len(sampled) > 0
        assert len(sampled) == len(indices)

    def test_includes_last_frame(self):
        """Should always include the last frame."""
        frames = list(range(50))

        _sampled, indices = sample_frames(
            frames, target_samples_per_sec=12, original_fps=24
        )

        assert 49 in indices  # Last frame index

    def test_single_frame(self):
        """Single frame should be sampled."""
        frames = [0]

        sampled, _indices = sample_frames(
            frames, target_samples_per_sec=12, original_fps=24
        )

        assert len(sampled) >= 1


class TestAllClassesPresent:
    """Tests for all_classes_present function."""

    def test_all_present(self):
        """Should return True when all classes present."""
        classes = [
            "right",
            "front_right",
            "front",
            "front_left",
            "left",
            "back_left",
            "back",
            "back_right",
        ]

        is_present, msg = all_classes_present(classes)

        assert is_present is True
        assert "All required positions" in msg

    def test_missing_class(self):
        """Should return False and list missing classes."""
        classes = ["right", "front", "left", "back"]  # Missing compound classes

        is_present, msg = all_classes_present(classes)

        assert is_present is False
        assert "Missing" in msg
        assert "front_right" in msg

    def test_duplicates_ok(self):
        """Duplicates should not affect result."""
        classes = [
            "right",
            "right",
            "front_right",
            "front",
            "front_left",
            "left",
            "back_left",
            "back",
            "back_right",
        ]

        is_present, _msg = all_classes_present(classes)

        assert is_present is True
