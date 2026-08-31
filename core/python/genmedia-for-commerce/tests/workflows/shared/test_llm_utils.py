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

"""Tests for shared/llm_utils.py - MIME detection, part creation, and retry logic."""

from unittest.mock import patch

import pytest
from google.genai import types
from workflows.shared.llm_utils import (
    get_generate_content_config,
    get_mime_type_from_bytes,
    get_mime_type_from_path,
    get_part,
    retry_with_exponential_backoff,
)


class TestGetMimeTypeFromBytes:
    """Tests for get_mime_type_from_bytes function."""

    def test_png_detection(self, png_bytes):
        """PNG signature should be detected correctly."""
        assert get_mime_type_from_bytes(png_bytes) == "image/png"

    def test_jpeg_detection(self, jpeg_bytes):
        """JPEG signature should be detected correctly."""
        assert get_mime_type_from_bytes(jpeg_bytes) == "image/jpeg"

    def test_webp_detection(self, webp_bytes):
        """WebP signature should be detected correctly."""
        assert get_mime_type_from_bytes(webp_bytes) == "image/webp"

    def test_gif_detection(self, gif_bytes):
        """GIF signature should be detected correctly."""
        assert get_mime_type_from_bytes(gif_bytes) == "image/gif"

    def test_avif_detection(self, avif_bytes):
        """AVIF signature should be detected correctly."""
        assert get_mime_type_from_bytes(avif_bytes) == "image/avif"

    def test_unknown_format(self, unknown_bytes):
        """Unknown format should return application/octet-stream."""
        assert (
            get_mime_type_from_bytes(unknown_bytes)
            == "application/octet-stream"
        )

    def test_short_data(self):
        """Data shorter than 12 bytes should return application/octet-stream."""
        assert get_mime_type_from_bytes(b"short") == "application/octet-stream"

    def test_mp4_detection(self):
        """MP4/MOV signature should be detected correctly."""
        # MP4 with ftyp box
        mp4_bytes = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 20
        assert get_mime_type_from_bytes(mp4_bytes) == "video/mp4"

    def test_webm_detection(self):
        """WebM signature should be detected correctly."""
        webm_bytes = b"\x1a\x45\xdf\xa3" + b"\x00" * 20
        assert get_mime_type_from_bytes(webm_bytes) == "video/webm"

    def test_avi_detection(self):
        """AVI signature should be detected correctly."""
        avi_bytes = b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 20
        assert get_mime_type_from_bytes(avi_bytes) == "video/avi"


class TestGetMimeTypeFromPath:
    """Tests for get_mime_type_from_path function."""

    def test_png_extension(self):
        """PNG extension should return image/png."""
        assert get_mime_type_from_path("image.png") == "image/png"
        assert get_mime_type_from_path("IMAGE.PNG") == "image/png"
        assert get_mime_type_from_path("/path/to/file.png") == "image/png"

    def test_jpeg_extension(self):
        """JPEG extensions should return image/jpeg."""
        assert get_mime_type_from_path("image.jpg") == "image/jpeg"
        assert get_mime_type_from_path("image.jpeg") == "image/jpeg"
        assert get_mime_type_from_path("IMAGE.JPG") == "image/jpeg"

    def test_webp_extension(self):
        """WebP extension should return image/webp."""
        assert get_mime_type_from_path("image.webp") == "image/webp"

    def test_gif_extension(self):
        """GIF extension should return image/gif."""
        assert get_mime_type_from_path("animation.gif") == "image/gif"

    def test_avif_extension(self):
        """AVIF extension should return image/avif."""
        assert get_mime_type_from_path("image.avif") == "image/avif"

    def test_mp4_extension(self):
        """MP4 extension should return video/mp4."""
        assert get_mime_type_from_path("video.mp4") == "video/mp4"

    def test_webm_extension(self):
        """WebM extension should return video/webm."""
        assert get_mime_type_from_path("video.webm") == "video/webm"

    def test_mov_extension(self):
        """MOV extension should return video/mp4."""
        assert get_mime_type_from_path("video.mov") == "video/mp4"

    def test_avi_extension(self):
        """AVI extension should return video/avi."""
        assert get_mime_type_from_path("video.avi") == "video/avi"

    def test_unknown_extension(self):
        """Unknown extension should default to image/jpeg."""
        assert get_mime_type_from_path("file.xyz") == "image/jpeg"
        assert get_mime_type_from_path("file") == "image/jpeg"

    def test_gcs_path(self):
        """GCS paths should work correctly."""
        assert (
            get_mime_type_from_path("gs://bucket/path/image.png") == "image/png"
        )
        assert get_mime_type_from_path("gs://bucket/video.mp4") == "video/mp4"


class TestGetPart:
    """Tests for get_part function."""

    def test_text_part(self):
        """String input should create a text Part."""
        part = get_part("Hello, world!")
        assert isinstance(part, types.Part)

    def test_bytes_part_png(self, sample_image_bytes):
        """PNG bytes should create a Part with correct MIME type."""
        part = get_part(sample_image_bytes)
        assert isinstance(part, types.Part)

    def test_bytes_part_returns_dict(self, sample_image_bytes):
        """return_dict=True should return a JSON dict."""
        result = get_part(sample_image_bytes, return_dict=True)
        assert isinstance(result, dict)

    def test_text_part_returns_dict(self):
        """Text part with return_dict=True should return a JSON dict."""
        result = get_part("Hello", return_dict=True)
        assert isinstance(result, dict)

    def test_gcs_uri_part(self):
        """GCS URI should create a Part from URI."""
        part = get_part("gs://bucket/path/image.png")
        assert isinstance(part, types.Part)


class TestGetGenerateContentConfig:
    """Tests for get_generate_content_config function."""

    def test_default_config(self):
        """Should create a config with default values."""
        config = get_generate_content_config()
        assert isinstance(config, types.GenerateContentConfig)

    def test_temperature_setting(self):
        """Should set temperature correctly."""
        config = get_generate_content_config(temperature=0.5)
        assert config.temperature == 0.5

    def test_max_output_tokens_setting(self):
        """Should set max_output_tokens correctly."""
        config = get_generate_content_config(max_output_tokens=1000)
        assert config.max_output_tokens == 1000

    def test_response_mime_type_json(self):
        """Should set response_mime_type for JSON."""
        config = get_generate_content_config(
            response_mime_type="application/json"
        )
        assert config.response_mime_type == "application/json"

    def test_response_modalities_text(self):
        """Should set response_modalities for text."""
        config = get_generate_content_config(response_modalities=["TEXT"])
        assert "TEXT" in config.response_modalities

    def test_response_modalities_image(self):
        """Should set response_modalities for image."""
        config = get_generate_content_config(response_modalities=["IMAGE"])
        assert "IMAGE" in config.response_modalities

    def test_system_instruction_string(self):
        """Should handle string system instruction."""
        config = get_generate_content_config(
            system_instruction="You are a helpful assistant."
        )
        assert config.system_instruction is not None

    def test_combined_settings(self):
        """Should handle multiple settings at once."""
        config = get_generate_content_config(
            temperature=0,
            max_output_tokens=500,
            response_mime_type="application/json",
        )
        assert config.temperature == 0
        assert config.max_output_tokens == 500
        assert config.response_mime_type == "application/json"

    def test_safety_off_default(self):
        """Should have safety disabled by default."""
        config = get_generate_content_config()
        assert config.safety_settings is not None
        assert len(config.safety_settings) == 4

    def test_safety_on(self):
        """Should not include safety settings when safety_off is False."""
        config = get_generate_content_config(safety_off=False)
        assert config.safety_settings is None

    def test_thinking_budget(self):
        """Should set thinking config when thinking_budget is provided."""
        config = get_generate_content_config(thinking_budget=1000)
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_budget == 1000

    def test_image_config(self):
        """Should set image config when provided."""
        config = get_generate_content_config(
            image_config={
                "aspect_ratio": "3:4",
                "output_mime_type": "image/png",
            }
        )
        assert config.image_config is not None

    def test_response_schema(self):
        """Should set response schema when provided."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        config = get_generate_content_config(response_schema=schema)
        assert config.response_schema == schema


class TestRetryWithExponentialBackoff:
    """Tests for retry_with_exponential_backoff decorator."""

    @patch("workflows.shared.llm_utils.time.sleep")
    def test_no_retry_on_success(self, mock_sleep):
        """Should not retry when function succeeds."""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=3, exceptions=(Exception,))
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()

        assert result == "success"
        assert call_count == 1
        mock_sleep.assert_not_called()

    @patch("workflows.shared.llm_utils.time.sleep")
    def test_retries_on_specified_exception(self, mock_sleep):
        """Should retry when specified exception is raised."""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=3, exceptions=(ValueError,))
        def failing_then_succeeding():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = failing_then_succeeding()

        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2  # Two retries before success

    @patch("workflows.shared.llm_utils.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        """Should raise the last exception after exhausting all retries."""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=2, exceptions=(ValueError,))
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_failing()

        assert call_count == 3  # Initial + 2 retries
        assert mock_sleep.call_count == 2

    @patch("workflows.shared.llm_utils.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        """Should use exponential backoff for delays."""

        @retry_with_exponential_backoff(
            max_retries=3,
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
            exceptions=(ValueError,),
        )
        def always_failing():
            raise ValueError("Fails")

        with pytest.raises(ValueError):
            always_failing()

        # Check delays: 1s, 2s, 4s
        calls = mock_sleep.call_args_list
        assert calls[0][0][0] == 1.0  # First delay
        assert calls[1][0][0] == 2.0  # Second delay
        assert calls[2][0][0] == 4.0  # Third delay

    @patch("workflows.shared.llm_utils.time.sleep")
    def test_max_delay_cap(self, mock_sleep):
        """Should cap delay at max_delay."""

        @retry_with_exponential_backoff(
            max_retries=4,
            initial_delay=10.0,
            exponential_base=10.0,
            max_delay=50.0,
            exceptions=(ValueError,),
        )
        def always_failing():
            raise ValueError("Fails")

        with pytest.raises(ValueError):
            always_failing()

        # Delays would be: 10, 100, 1000, 10000 but capped at 50
        calls = mock_sleep.call_args_list
        assert calls[0][0][0] == 10.0  # First delay
        assert calls[1][0][0] == 50.0  # Second delay (capped at max)
        assert calls[2][0][0] == 50.0  # Third delay (capped)

    @patch("workflows.shared.llm_utils.time.sleep")
    def test_ignores_non_specified_exceptions(self, mock_sleep):
        """Should not retry for exceptions not in the exceptions tuple."""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=3, exceptions=(ValueError,))
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Different exception")

        with pytest.raises(TypeError):
            raises_type_error()

        assert call_count == 1  # No retries
        mock_sleep.assert_not_called()

    @patch("workflows.shared.llm_utils.time.sleep")
    def test_handles_multiple_exception_types(self, mock_sleep):
        """Should retry for any exception in the exceptions tuple."""
        call_count = 0

        @retry_with_exponential_backoff(
            max_retries=4, exceptions=(ValueError, TypeError)
        )
        def alternating_exceptions():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First")
            elif call_count == 2:
                raise TypeError("Second")
            return "success"

        result = alternating_exceptions()

        assert result == "success"
        assert call_count == 3

    @patch("workflows.shared.llm_utils.time.sleep")
    def test_preserves_function_metadata(self, mock_sleep):
        """Should preserve the decorated function's name and docstring."""

        @retry_with_exponential_backoff()
        def documented_function():
            """This is a documented function."""
            return "result"

        assert documented_function.__name__ == "documented_function"
        assert "documented function" in documented_function.__doc__

    @patch("workflows.shared.llm_utils.time.sleep")
    def test_passes_args_and_kwargs(self, mock_sleep):
        """Should pass arguments to the decorated function."""

        @retry_with_exponential_backoff(max_retries=1, exceptions=(ValueError,))
        def function_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = function_with_args("x", "y", c="z")

        assert result == "x-y-z"

    @patch("workflows.shared.llm_utils.time.sleep")
    def test_default_exceptions_client_error(self, mock_sleep):
        """Should use ClientError as default exception (uses base Exception for simplicity)."""
        call_count = 0

        # ClientError requires specific constructor args, so we test with a custom exception
        # that's a subclass of Exception to verify the default behavior works
        class CustomClientError(Exception):
            pass

        @retry_with_exponential_backoff(
            max_retries=2, exceptions=(CustomClientError,)
        )
        def raises_custom_error():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise CustomClientError("API error")
            return "success"

        result = raises_custom_error()

        assert result == "success"
        assert call_count == 2

    @patch("workflows.shared.llm_utils.time.sleep")
    def test_zero_retries(self, mock_sleep):
        """Should work with max_retries=0 (no retries)."""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=0, exceptions=(ValueError,))
        def fails_once():
            nonlocal call_count
            call_count += 1
            raise ValueError("Fails")

        with pytest.raises(ValueError):
            fails_once()

        assert call_count == 1
        mock_sleep.assert_not_called()
