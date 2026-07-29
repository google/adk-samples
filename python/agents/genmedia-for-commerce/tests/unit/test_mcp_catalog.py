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

"""Unit tests for the catalog MCP tool (catalog_mcp.py)."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# catalog_mcp imports vector_search, which calls _load() at import time.
# We must patch vector_search before importing catalog_mcp.
# ---------------------------------------------------------------------------

_VS_MODULE = "workflows.shared.vector_search"
_FAKE_EMBEDDINGS = np.zeros((5, 4), dtype=np.float32)


@pytest.fixture(autouse=True)
def _patch_vector_search_import():
    """Prevent vector_search from performing real I/O at import time."""
    with (
        patch(f"{_VS_MODULE}.np.load", return_value=_FAKE_EMBEDDINGS),
        patch(f"{_VS_MODULE}.pq.read_table", return_value=MagicMock()),
        patch(f"{_VS_MODULE}.genai.Client", return_value=MagicMock()),
    ):
        # Force re-import of vector_search so _load() runs under patches
        if _VS_MODULE in sys.modules:
            del sys.modules[_VS_MODULE]

        import workflows.shared.vector_search as vs

        vs._embeddings = _FAKE_EMBEDDINGS
        vs._metadata_table = MagicMock()
        vs._embed_client = MagicMock()
        yield


# Now we can safely import catalog_mcp (it will use the patched vector_search)
from mcp_server.shared.catalog.catalog_mcp import run_catalog_search  # noqa: E402


class TestRunCatalogSearch:
    """Tests for run_catalog_search."""

    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, _patch_vector_search_import):
        fake_results = [
            {
                "id": "item-1",
                "data": {
                    "description": "A red summer dress",
                    "img_path": "gs://bucket/img_1.png",
                    "category": "dress",
                    "color": "red",
                    "style": "casual",
                    "audience": "women",
                },
                "score": 0.95,
            },
            {
                "id": "item-2",
                "data": {
                    "description": "Blue denim jacket",
                    "img_path": "gs://bucket/img_2.png",
                    "category": "jacket",
                    "color": "blue",
                    "style": "streetwear",
                    "audience": "men",
                },
                "score": 0.88,
            },
        ]

        with patch(
            "mcp_server.shared.catalog.catalog_mcp.search", return_value=fake_results
        ):
            result = await run_catalog_search(query="red dress", k=5)

        assert "results" in result
        assert len(result["results"]) == 2

        first = result["results"][0]
        assert first["description"] == "A red summer dress"
        assert first["img_path"] == "gs://bucket/img_1.png"
        assert first["category"] == "dress"
        assert first["color"] == "red"
        assert first["style"] == "casual"
        assert first["audience"] == "women"

    @pytest.mark.asyncio
    async def test_img_path_preserved(self, _patch_vector_search_import):
        """img_path should be passed through as-is from the data dict."""
        fake_results = [
            {
                "id": "item-1",
                "data": {
                    "description": "test",
                    "img_path": "gs://mybucket/path/image.png",
                    "category": "",
                    "color": "",
                    "style": "",
                    "audience": "",
                },
                "score": 0.5,
            },
        ]

        with patch(
            "mcp_server.shared.catalog.catalog_mcp.search", return_value=fake_results
        ):
            result = await run_catalog_search(query="test")

        assert result["results"][0]["img_path"] == "gs://mybucket/path/image.png"

    @pytest.mark.asyncio
    async def test_search_error_returns_empty_results_with_error(
        self, _patch_vector_search_import
    ):
        """When search raises, should return empty results and error message."""
        with patch(
            "mcp_server.shared.catalog.catalog_mcp.search",
            side_effect=RuntimeError("connection timeout"),
        ):
            result = await run_catalog_search(query="failing query")

        assert result["results"] == []
        assert "error" in result
        assert "connection timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_search_results(self, _patch_vector_search_import):
        """Empty search results should produce empty formatted list."""
        with patch("mcp_server.shared.catalog.catalog_mcp.search", return_value=[]):
            result = await run_catalog_search(query="nothing matches")

        assert result["results"] == []
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_default_k_is_12(self, _patch_vector_search_import):
        """Default k parameter should be 12."""
        with patch("mcp_server.shared.catalog.catalog_mcp.search", return_value=[]):
            await run_catalog_search(query="test")

        # run_in_executor passes (None, search, query, k) -- check the call
        # Since search is called via run_in_executor, we check mock_search wasn't
        # called directly but via the executor. We verify the function signature instead.
        import inspect

        sig = inspect.signature(run_catalog_search)
        assert sig.parameters["k"].default == 12

    @pytest.mark.asyncio
    async def test_missing_data_fields_return_empty_dict(
        self, _patch_vector_search_import
    ):
        """Missing data dict should return an empty dict."""
        fake_results = [
            {
                "id": "item-1",
                "data": {},  # all fields missing
                "score": 0.1,
            },
        ]

        with patch(
            "mcp_server.shared.catalog.catalog_mcp.search", return_value=fake_results
        ):
            result = await run_catalog_search(query="sparse data")

        item = result["results"][0]
        assert item == {}
