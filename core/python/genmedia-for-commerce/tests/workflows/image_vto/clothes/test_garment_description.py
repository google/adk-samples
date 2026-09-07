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

"""Tests for garment_description.py - Garment description with mocked Gemini."""

import json
from unittest.mock import Mock, patch

import pytest
from workflows.image_vto.clothes.garment_description import (
    describe_all_garments,
    describe_garment_for_vto,
)


@pytest.fixture
def mock_client():
    """Create a mock Gemini client."""
    return Mock()


@pytest.fixture
def garment_image_bytes():
    """Minimal garment image bytes for testing."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


class TestDescribeGarmentForVto:
    """Tests for describe_garment_for_vto function."""

    @patch("workflows.image_vto.clothes.garment_description.generate_gemini")
    def test_parses_valid_json_response(
        self, mock_generate, mock_client, garment_image_bytes
    ):
        """Should parse a valid JSON response into general and details keys."""
        mock_generate.return_value = json.dumps(
            {
                "general": "Nike running shorts, black polyester",
                "details": "[EXTERIOR] White Nike swoosh on left thigh\n[INTERIOR] Size tag at waistband",
            }
        )

        result = describe_garment_for_vto(mock_client, garment_image_bytes)

        assert result["general"] == "Nike running shorts, black polyester"
        assert "[EXTERIOR]" in result["details"]
        assert "[INTERIOR]" in result["details"]

    @patch("workflows.image_vto.clothes.garment_description.generate_gemini")
    def test_handles_missing_keys_in_response(
        self, mock_generate, mock_client, garment_image_bytes
    ):
        """Should default to empty strings when JSON keys are missing."""
        mock_generate.return_value = json.dumps({"other_key": "value"})

        result = describe_garment_for_vto(mock_client, garment_image_bytes)

        assert result["general"] == ""
        assert result["details"] == ""

    @patch("workflows.image_vto.clothes.garment_description.generate_gemini")
    def test_returns_fallback_on_json_parse_error(
        self, mock_generate, mock_client, garment_image_bytes
    ):
        """Should return empty fallback dict when Gemini returns invalid JSON."""
        mock_generate.return_value = "This is not JSON at all"

        result = describe_garment_for_vto(mock_client, garment_image_bytes)

        assert result == {"general": "", "details": ""}

    @patch("workflows.image_vto.clothes.garment_description.generate_gemini")
    def test_returns_fallback_on_api_exception(
        self, mock_generate, mock_client, garment_image_bytes
    ):
        """Should return empty fallback dict when generate_gemini raises."""
        mock_generate.side_effect = RuntimeError("API timeout")

        result = describe_garment_for_vto(mock_client, garment_image_bytes)

        assert result == {"general": "", "details": ""}

    @patch("workflows.image_vto.clothes.garment_description.generate_gemini")
    def test_passes_image_and_client_to_gemini(
        self, mock_generate, mock_client, garment_image_bytes
    ):
        """Should pass the garment image and client to generate_gemini."""
        mock_generate.return_value = json.dumps({"general": "", "details": ""})

        describe_garment_for_vto(
            mock_client, garment_image_bytes, model="gemini-custom"
        )

        call_kwargs = mock_generate.call_args.kwargs
        assert garment_image_bytes in call_kwargs["text_images_pieces"]
        assert call_kwargs["client"] is mock_client
        assert call_kwargs["model"] == "gemini-custom"


class TestDescribeAllGarments:
    """Tests for describe_all_garments function."""

    @patch(
        "workflows.image_vto.clothes.garment_description.describe_garment_for_vto"
    )
    def test_returns_empty_list_for_no_images(self, mock_describe, mock_client):
        """Should return empty list when given no garment images."""
        result = describe_all_garments(mock_client, [])

        assert result == []
        mock_describe.assert_not_called()

    @patch(
        "workflows.image_vto.clothes.garment_description.describe_garment_for_vto"
    )
    def test_describes_garments_in_parallel(
        self, mock_describe, mock_client, garment_image_bytes
    ):
        """Should describe all garments in parallel, returning one dict per image."""
        mock_describe.side_effect = [
            {"general": "Shirt", "details": "[EXTERIOR] Logo"},
            {"general": "Pants", "details": "[INTERIOR] Tag"},
        ]

        images = [garment_image_bytes, garment_image_bytes]
        result = describe_all_garments(mock_client, images)

        assert len(result) == 2
        assert result[0]["general"] == "Shirt"
        assert result[1]["general"] == "Pants"
        assert mock_describe.call_count == 2
