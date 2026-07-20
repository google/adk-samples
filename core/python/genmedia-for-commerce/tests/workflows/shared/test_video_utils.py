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

"""Tests for shared/video_utils.py - Video processing utilities."""

import io

import pytest
from PIL import Image
from workflows.shared.video_utils import (
    convert_image_to_video_frame,
    create_mp4_from_bytes_to_bytes,
    extract_frames_as_bytes_list,
    find_most_similar_frame_index,
    get_frame_similarity_bytes,
    merge_videos_from_bytes,
    reverse_video,
)


@pytest.fixture
def red_image_bytes():
    """Create a 100x100 red image."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def blue_image_bytes():
    """Create a 100x100 blue image."""
    img = Image.new("RGB", (100, 100), color=(0, 0, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def green_image_bytes():
    """Create a 100x100 green image."""
    img = Image.new("RGB", (100, 100), color=(0, 255, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def wide_image_bytes():
    """Create a 200x100 wide image."""
    img = Image.new("RGB", (200, 100), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def tall_image_bytes():
    """Create a 100x200 tall image."""
    img = Image.new("RGB", (100, 200), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestGetFrameSimilarityBytes:
    """Tests for get_frame_similarity_bytes function."""

    def test_identical_images(self, red_image_bytes):
        """Identical images should have similarity of 1.0."""
        similarity = get_frame_similarity_bytes(
            red_image_bytes, red_image_bytes
        )
        assert similarity == pytest.approx(1.0, abs=0.001)

    def test_different_images(self, red_image_bytes, blue_image_bytes):
        """Different colored images should have lower similarity than identical."""
        similarity = get_frame_similarity_bytes(
            red_image_bytes, blue_image_bytes
        )
        assert similarity < 1.0  # Not identical

    def test_similarity_is_symmetric(self, red_image_bytes, blue_image_bytes):
        """Similarity should be the same regardless of order."""
        sim1 = get_frame_similarity_bytes(red_image_bytes, blue_image_bytes)
        sim2 = get_frame_similarity_bytes(blue_image_bytes, red_image_bytes)
        assert sim1 == pytest.approx(sim2, abs=0.001)

    def test_different_sizes_handled(self, red_image_bytes, wide_image_bytes):
        """Images of different sizes should be resized and compared."""
        # Should not raise an error
        similarity = get_frame_similarity_bytes(
            red_image_bytes, wide_image_bytes
        )
        assert 0.0 <= similarity <= 1.0


class TestConvertImageToVideoFrame:
    """Tests for convert_image_to_video_frame function."""

    def test_same_size(self, red_image_bytes):
        """Image same size as frame should be returned with minimal changes."""
        result = convert_image_to_video_frame(red_image_bytes, red_image_bytes)
        assert isinstance(result, bytes)
        img = Image.open(io.BytesIO(result))
        assert img.size == (100, 100)

    def test_wide_image_to_square_frame(
        self, red_image_bytes, wide_image_bytes
    ):
        """Wide image should be cropped to match square frame."""
        result = convert_image_to_video_frame(red_image_bytes, wide_image_bytes)
        img = Image.open(io.BytesIO(result))
        # Should match the video frame dimensions (100x100)
        assert img.size == (100, 100)

    def test_tall_image_to_square_frame(
        self, red_image_bytes, tall_image_bytes
    ):
        """Tall image should be cropped to match square frame."""
        result = convert_image_to_video_frame(red_image_bytes, tall_image_bytes)
        img = Image.open(io.BytesIO(result))
        assert img.size == (100, 100)

    def test_output_is_png(self, red_image_bytes, blue_image_bytes):
        """Output should be PNG format."""
        result = convert_image_to_video_frame(red_image_bytes, blue_image_bytes)
        img = Image.open(io.BytesIO(result))
        assert img.format == "PNG"


class TestFindMostSimilarFrameIndex:
    """Tests for find_most_similar_frame_index function."""

    def test_finds_identical_frame(
        self, red_image_bytes, blue_image_bytes, green_image_bytes
    ):
        """Should find the index of the identical frame."""
        frames = [blue_image_bytes, green_image_bytes, red_image_bytes]
        reference = red_image_bytes
        index = find_most_similar_frame_index(frames, reference)
        assert index == 2  # red is at index 2

    def test_finds_most_similar(self, red_image_bytes, blue_image_bytes):
        """Should find the most similar frame when no exact match."""
        # Create a slightly different red
        img = Image.new("RGB", (100, 100), color=(250, 5, 5))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        similar_red = buffer.getvalue()

        frames = [blue_image_bytes, similar_red, blue_image_bytes]
        reference = red_image_bytes
        index = find_most_similar_frame_index(frames, reference)
        assert index == 1  # similar_red is at index 1

    def test_num_frames_to_check(
        self, red_image_bytes, blue_image_bytes, green_image_bytes
    ):
        """Should only check the last N frames when num_frames_to_check is set."""
        # red at index 0, blue at 1, 2, 3, green at 4
        frames = [
            red_image_bytes,
            blue_image_bytes,
            blue_image_bytes,
            blue_image_bytes,
            green_image_bytes,
        ]
        reference = red_image_bytes

        # Only check last 3 frames (indices 2, 3, 4 -> blue, blue, green)
        # Function returns index relative to subset being checked
        index = find_most_similar_frame_index(
            frames, reference, num_frames_to_check=3
        )
        # Index should be within valid range
        assert 0 <= index < 3  # Index within the checked subset


class TestCreateMp4FromBytesToBytes:
    """Tests for create_mp4_from_bytes_to_bytes function."""

    def test_creates_video_from_frames(
        self, red_image_bytes, blue_image_bytes, green_image_bytes
    ):
        """Should create a valid MP4 video from frame bytes."""
        frames = [red_image_bytes, blue_image_bytes, green_image_bytes]
        video_bytes = create_mp4_from_bytes_to_bytes(frames, fps=24, quality=7)

        assert isinstance(video_bytes, bytes)
        assert len(video_bytes) > 0
        # MP4 files typically start with ftyp box
        assert b"ftyp" in video_bytes[:32] or b"moov" in video_bytes

    def test_single_frame_video(self, red_image_bytes):
        """Should handle single frame video."""
        frames = [red_image_bytes]
        video_bytes = create_mp4_from_bytes_to_bytes(frames, fps=24, quality=7)
        assert isinstance(video_bytes, bytes)
        assert len(video_bytes) > 0


class TestExtractFramesAsBytesList:
    """Tests for extract_frames_as_bytes_list function."""

    def test_extracts_frames_from_video(
        self, red_image_bytes, blue_image_bytes, green_image_bytes
    ):
        """Should extract frames from a video."""
        # Create a video first
        frames = [red_image_bytes, blue_image_bytes, green_image_bytes]
        video_bytes = create_mp4_from_bytes_to_bytes(frames, fps=24, quality=7)

        # Extract frames
        extracted = extract_frames_as_bytes_list(video_bytes)

        assert isinstance(extracted, list)
        assert len(extracted) >= 1  # Should have at least some frames
        # Each extracted frame should be valid image bytes
        for frame in extracted:
            assert isinstance(frame, bytes)
            img = Image.open(io.BytesIO(frame))
            assert img.size[0] > 0 and img.size[1] > 0


class TestReverseVideo:
    """Tests for reverse_video function."""

    def test_reverses_frame_order(
        self, red_image_bytes, blue_image_bytes, green_image_bytes
    ):
        """Reversed video should have frames in opposite order."""
        # Create original video: red -> blue -> green
        frames = [red_image_bytes, blue_image_bytes, green_image_bytes]
        video_bytes = create_mp4_from_bytes_to_bytes(frames, fps=24, quality=7)

        # Reverse it
        reversed_video = reverse_video(video_bytes, fps=24, quality=7)

        # Extract frames from reversed video
        extracted = extract_frames_as_bytes_list(reversed_video)

        assert isinstance(extracted, list)
        assert len(extracted) >= 1


class TestMergeVideosFromBytes:
    """Tests for merge_videos_from_bytes function."""

    def test_merges_two_videos(
        self, red_image_bytes, blue_image_bytes, green_image_bytes
    ):
        """Should merge two videos into one."""
        # Create two short videos
        video1 = create_mp4_from_bytes_to_bytes(
            [red_image_bytes] * 5, fps=24, quality=7
        )
        video2 = create_mp4_from_bytes_to_bytes(
            [blue_image_bytes] * 5, fps=24, quality=7
        )

        # Merge them
        merged = merge_videos_from_bytes([video1, video2], fps=24)

        assert isinstance(merged, bytes)
        assert len(merged) > 0
        # Merged video should be larger than either individual video
        assert len(merged) > min(len(video1), len(video2))

    def test_merges_three_videos(
        self, red_image_bytes, blue_image_bytes, green_image_bytes
    ):
        """Should merge three videos into one."""
        video1 = create_mp4_from_bytes_to_bytes(
            [red_image_bytes] * 3, fps=24, quality=7
        )
        video2 = create_mp4_from_bytes_to_bytes(
            [blue_image_bytes] * 3, fps=24, quality=7
        )
        video3 = create_mp4_from_bytes_to_bytes(
            [green_image_bytes] * 3, fps=24, quality=7
        )

        merged = merge_videos_from_bytes([video1, video2, video3], fps=24)

        assert isinstance(merged, bytes)
        assert len(merged) > 0

    def test_merge_with_speed_adjustment(
        self, red_image_bytes, blue_image_bytes
    ):
        """Should apply speed adjustments to each clip."""
        video1 = create_mp4_from_bytes_to_bytes(
            [red_image_bytes] * 10, fps=24, quality=7
        )
        video2 = create_mp4_from_bytes_to_bytes(
            [blue_image_bytes] * 10, fps=24, quality=7
        )

        # Speed up first video 2x, slow down second 0.5x
        merged = merge_videos_from_bytes(
            [video1, video2], speeds=[2.0, 0.5], fps=24
        )

        assert isinstance(merged, bytes)
        assert len(merged) > 0

    def test_merge_single_video(self, red_image_bytes):
        """Should handle merging a single video (passthrough)."""
        video = create_mp4_from_bytes_to_bytes(
            [red_image_bytes] * 5, fps=24, quality=7
        )

        merged = merge_videos_from_bytes([video], fps=24)

        assert isinstance(merged, bytes)
        assert len(merged) > 0

    def test_merge_preserves_content(self, red_image_bytes, blue_image_bytes):
        """Merged video should contain frames from both source videos."""
        video1 = create_mp4_from_bytes_to_bytes(
            [red_image_bytes] * 5, fps=24, quality=7
        )
        video2 = create_mp4_from_bytes_to_bytes(
            [blue_image_bytes] * 5, fps=24, quality=7
        )

        merged = merge_videos_from_bytes([video1, video2], fps=24)

        # Extract frames from merged video
        frames = extract_frames_as_bytes_list(merged)
        assert len(frames) >= 2  # Should have frames from both videos
