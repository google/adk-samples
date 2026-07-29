# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the product_fitting MCP tool."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# We test the tool wrapper function, not the full pipeline
from mcp_server.product_enrichment.product_fitting.product_fitting_mcp import (
    AVAILABLE_PRESETS,
    _load_preset_model_photos,
    run_product_fitting,
)


class TestLoadPresetModelPhotos:
    """Tests for _load_preset_model_photos."""

    def test_available_presets_discovered(self):
        """Model presets should be discovered at import time."""
        assert len(AVAILABLE_PRESETS) > 0
        assert "european_woman" in AVAILABLE_PRESETS
        assert "european_man" in AVAILABLE_PRESETS

    def test_load_valid_preset(self):
        """Loading a valid preset should return front_top and front_bottom photos."""
        photos = _load_preset_model_photos("european", "woman")
        assert "front_top" in photos
        assert "front_bottom" in photos
        assert isinstance(photos["front_top"], bytes)
        assert len(photos["front_top"]) > 0

    def test_load_with_gender_alias(self):
        """Gender aliases like 'female' should map to 'woman'."""
        photos = _load_preset_model_photos("european", "female")
        assert "front_top" in photos

    def test_invalid_ethnicity_raises(self):
        """Invalid ethnicity should raise ValueError with available presets."""
        with pytest.raises(ValueError, match="No model preset"):
            _load_preset_model_photos("martian", "woman")

    def test_invalid_gender_raises(self):
        """Invalid gender should raise ValueError."""
        with pytest.raises(ValueError, match="No model preset"):
            _load_preset_model_photos("european", "robot")


class TestRunProductFitting:
    """Tests for the run_product_fitting async function."""

    @pytest.mark.asyncio
    async def test_empty_garment_list_returns_error(self):
        """Should return error when no garment images provided."""
        result = await run_product_fitting(
            garment_images_base64=[],
            gender="woman",
        )
        assert "error" in result
        assert "garment" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_base64_returns_error(self):
        """Should return error for invalid base64 encoding."""
        result = await run_product_fitting(
            garment_images_base64=["not-valid-base64!!!"],
            gender="woman",
        )
        assert "error" in result
        assert "base64" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_ethnicity_returns_error(self):
        """Should return error for invalid ethnicity preset."""
        # Create a tiny valid base64 image (1x1 white pixel PNG)
        tiny_png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        ).decode()

        result = await run_product_fitting(
            garment_images_base64=[tiny_png],
            gender="woman",
            ethnicity="martian",
        )
        assert "error" in result
        assert "preset" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_call_delegates_to_pipeline(self):
        """Should call run_fitting_pipeline with correct args."""
        tiny_png = base64.b64encode(b"\x89PNG_fake_image_data").decode()

        mock_result = {
            "front": {
                "image": b"front_image_bytes",
                "status": "ready",
                "validation": {"garments_score": 85},
                "total_attempts": 2,
            },
            "back": None,
        }

        with (
            patch(
                "mcp_server.product_enrichment.product_fitting.product_fitting_mcp.run_fitting_pipeline",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as mock_pipeline,
            patch(
                "mcp_server.product_enrichment.product_fitting.product_fitting_mcp._get_clients",
                return_value=(MagicMock(), MagicMock()),
            ),
        ):
            result = await run_product_fitting(
                garment_images_base64=[tiny_png],
                gender="woman",
                ethnicity="european",
            )

            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["gender"] == "woman"
            assert len(call_kwargs["garment_images_bytes"]) == 1
            assert "front_top" in call_kwargs["model_photo_map"]

            assert result["front"] is not None
            assert result["front"]["status"] == "ready"
            assert result["back"] is None


class TestMCPServerTool:
    """Tests for the MCP server tool registration."""

    def test_server_imports(self):
        """MCP server module should import without errors."""
        import mcp_server.server

        assert hasattr(mcp_server.server, "server")
        assert hasattr(mcp_server.server, "product_fitting")
