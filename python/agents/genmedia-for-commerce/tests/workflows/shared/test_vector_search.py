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

"""Unit tests for workflows.shared.vector_search."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# The vector_search module calls _load() at import time, which downloads from
# GCS, loads .npy / .parquet files, and creates a genai client.  We must patch
# all of that BEFORE the first import.
# ---------------------------------------------------------------------------

_VS_MODULE = "workflows.shared.vector_search"

# Build small synthetic data: 5 vectors of dimension 4
_DIM = 4
_NUM = 5
_FAKE_EMBEDDINGS = (
    np.random.default_rng(42).standard_normal((_NUM, _DIM)).astype(np.float32)
)
_FAKE_CLIENT = MagicMock()


def _make_fake_metadata(n: int):
    """Return a mock pyarrow table with n rows of fake catalogue data."""
    table = MagicMock()

    def _take(indices):
        result = MagicMock()
        result.to_pydict.return_value = {
            "idx": [f"item-{i}" for i in indices],
            "description": [f"desc-{i}" for i in indices],
            "img_path": [f"gs://bucket/img_{i}.png" for i in indices],
            "style": [f"style-{i}" for i in indices],
            "audience": [f"audience-{i}" for i in indices],
            "season": [f"season-{i}" for i in indices],
            "category": [f"category-{i}" for i in indices],
            "color": [f"color-{i}" for i in indices],
        }
        return result

    table.take = _take
    return table


_FAKE_METADATA = _make_fake_metadata(_NUM)


@pytest.fixture(autouse=True)
def _patch_vector_search_load():
    """Patch heavy I/O so the module can be imported safely."""
    with (
        patch(f"{_VS_MODULE}.np.load", return_value=_FAKE_EMBEDDINGS),
        patch(f"{_VS_MODULE}.pq.read_table", return_value=_FAKE_METADATA),
        patch(f"{_VS_MODULE}.genai.Client", return_value=_FAKE_CLIENT),
    ):
        # Force re-import so _load() runs under our patches
        if _VS_MODULE in sys.modules:
            del sys.modules[_VS_MODULE]

        import workflows.shared.vector_search as vs

        # Directly set module-level state so tests see the fakes
        vs._embeddings = _FAKE_EMBEDDINGS
        vs._metadata_table = _FAKE_METADATA
        vs._embed_client = _FAKE_CLIENT

        yield vs


# ---- Tests ----------------------------------------------------------------


class TestSearchByVector:
    """Tests for search_by_vector (dot-product similarity)."""

    def test_returns_top_k_results(self, _patch_vector_search_load):
        vs = _patch_vector_search_load
        query = np.ones(_DIM, dtype=np.float32)
        results = vs.search_by_vector(query.tolist(), k=3)
        assert len(results) == 3

    def test_results_sorted_descending_by_score(self, _patch_vector_search_load):
        vs = _patch_vector_search_load
        query = np.ones(_DIM, dtype=np.float32)
        results = vs.search_by_vector(query.tolist(), k=_NUM)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_correct_top_result(self, _patch_vector_search_load):
        """The vector with highest dot product should be first."""
        vs = _patch_vector_search_load
        query = _FAKE_EMBEDDINGS[2]  # pick a specific row as query
        results = vs.search_by_vector(query.tolist(), k=1)

        # Compute expected best index manually
        scores = _FAKE_EMBEDDINGS @ query
        expected_idx = int(np.argmax(scores))
        assert results[0]["id"] == f"item-{expected_idx}"

    def test_result_structure(self, _patch_vector_search_load):
        vs = _patch_vector_search_load
        query = np.ones(_DIM, dtype=np.float32)
        results = vs.search_by_vector(query.tolist(), k=1)
        r = results[0]
        assert "id" in r
        assert "data" in r
        assert "score" in r
        assert "description" in r["data"]
        assert "category" in r["data"]
        assert "img_path" in r["data"]

    def test_k_larger_than_catalogue(self, _patch_vector_search_load):
        """Requesting more results than items should return all items."""
        vs = _patch_vector_search_load
        query = np.ones(_DIM, dtype=np.float32)
        results = vs.search_by_vector(query.tolist(), k=100)
        assert len(results) == _NUM


class TestSearchForOutfitItem:
    """Tests for search_for_outfit_item."""

    def test_enriches_item_with_matched_products(self, _patch_vector_search_load):
        vs = _patch_vector_search_load
        item = {"description": "red dress", "style": "casual"}

        with patch.object(
            vs,
            "search",
            return_value=[
                {"id": "p1", "data": {"description": "A red dress"}, "score": 0.95},
                {"id": "p2", "data": {"description": "Crimson gown"}, "score": 0.88},
            ],
        ):
            result = vs.search_for_outfit_item(item, k=5)

        assert "matched_products" in result
        assert len(result["matched_products"]) == 2
        # Original keys preserved
        assert result["description"] == "red dress"
        assert result["style"] == "casual"

    def test_search_error_populates_error_field(self, _patch_vector_search_load):
        vs = _patch_vector_search_load
        item = {"description": "blue jacket"}

        with patch.object(vs, "search", side_effect=RuntimeError("embedding failed")):
            result = vs.search_for_outfit_item(item, k=5)

        assert result["matched_products"] == []
        assert "search_error" in result
        assert "embedding failed" in result["search_error"]

    def test_passes_k_to_search(self, _patch_vector_search_load):
        vs = _patch_vector_search_load
        item = {"description": "test"}

        with patch.object(vs, "search", return_value=[]) as mock_search:
            vs.search_for_outfit_item(item, k=42)
            mock_search.assert_called_once_with("test", k=42)


class TestSearch:
    """Tests for the high-level search() function."""

    def test_calls_embed_and_search_by_vector(self, _patch_vector_search_load):
        vs = _patch_vector_search_load
        fake_embedding = [0.1, 0.2, 0.3, 0.4]
        with (
            patch.object(vs, "embed_query", return_value=fake_embedding) as mock_embed,
            patch.object(
                vs, "search_by_vector", return_value=[{"id": "x"}]
            ) as mock_sbv,
        ):
            results = vs.search("hello", k=7)
            mock_embed.assert_called_once_with("hello")
            mock_sbv.assert_called_once_with(embedding=fake_embedding, k=7)
            assert results == [{"id": "x"}]


class TestEmbedQuery:
    """Tests for embed_query."""

    def test_returns_list_of_floats(self, _patch_vector_search_load):
        vs = _patch_vector_search_load
        fake_arr = np.array([0.1, 0.2, 0.3])
        with patch.object(vs, "embed_gemini", return_value=fake_arr):
            result = vs.embed_query("test query")
            assert result == pytest.approx([0.1, 0.2, 0.3])
            assert isinstance(result, list)
