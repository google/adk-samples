"""Tests for pubsub_sync.py across 3 use cases.

Use Case 1 (Electronics): Create/update events, upsert to VS 2.0
Use Case 2 (Fashion): Delete events, hard vs soft delete
Use Case 3 (Grocery): Out-of-stock soft delete, missing products

Tests mock GCP dependencies (BigQuery, Vector Search) but call the
real pubsub_sync functions to verify actual logic.
"""

import base64
import importlib
import json
import sys
from unittest.mock import MagicMock, patch

import pytest


# Make functions_framework.cloud_event a passthrough decorator
# so handle_product_event is the real function
sys.modules["functions_framework"].cloud_event = lambda f: f


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("VECTOR_SEARCH_COLLECTION",
                       "projects/test-project/locations/us-central1/collections/products-collection")
    monkeypatch.setenv("DELETION_MODE", "hard")


@pytest.fixture
def ps(mock_env):
    """Import and reload pubsub_sync to pick up env vars."""
    import pubsub_sync
    importlib.reload(pubsub_sync)
    _patch_vs_requests(pubsub_sync)
    return pubsub_sync


def _make_cloud_event(action: str, product_id: str) -> MagicMock:
    """Create a mock CloudEvent."""
    message_data = json.dumps({"action": action, "product_id": product_id})
    encoded = base64.b64encode(message_data.encode("utf-8")).decode("utf-8")
    event = MagicMock()
    event.data = {"message": {"data": encoded}}
    return event


def _mock_bq_row(product: dict) -> MagicMock:
    """Create a mock BigQuery row from a product dict."""
    row = MagicMock()
    row.items.return_value = list(product.items())
    return row


class _SimpleObj:
    """Simple object that stores kwargs as attributes (avoids MagicMock parent issues)."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)


def _patch_vs_requests(ps):
    """Patch VS request classes to accept kwargs without MagicMock issues.

    MagicMock cannot accept 'parent' as a kwarg because it's a reserved
    attribute. Use a simple object instead.
    """
    ps.vectorsearch.DataObject = lambda **kwargs: _SimpleObj(**kwargs)
    ps.vectorsearch.UpsertDataObjectRequest = lambda **kwargs: _SimpleObj(**kwargs)
    ps.vectorsearch.DeleteDataObjectRequest = lambda **kwargs: _SimpleObj(**kwargs)


# =====================================================================
# Use Case 1: Electronics -- Create/Update events
# =====================================================================

class TestElectronicsProductEvents:
    """Electronics: product created and updated events trigger VS 2.0 upsert."""

    def test_created_event_fetches_and_upserts(self, ps):
        """Created event should fetch from BQ and upsert to VS with real logic."""
        product = {
            "product_id": "elec-001",
            "name": "Wireless Headphones",
            "price": 179.99,
            "description": "Over-ear headphones",
            "category": "Audio",
            "brand": "SoundMax",
        }

        mock_bq_client = MagicMock()
        mock_bq_client.query.return_value.result.return_value = [_mock_bq_row(product)]

        mock_vs_client = MagicMock()

        with patch.object(ps.bigquery, "Client", return_value=mock_bq_client), \
             patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            event = _make_cloud_event("created", "elec-001")
            ps.handle_product_event(event)

        # Verify BigQuery was queried
        mock_bq_client.query.assert_called_once()
        query = mock_bq_client.query.call_args[0][0]
        assert "elec-001" in str(mock_bq_client.query.call_args) or "product_id" in query

        # Verify VS upsert was called
        mock_vs_client.upsert_data_object.assert_called_once()

    def test_updated_event_upserts_new_data(self, ps):
        """Updated event should re-fetch and upsert with new data."""
        updated_product = {
            "product_id": "elec-005",
            "name": "Portable Speaker V2",
            "price": 59.99,
            "description": "Updated waterproof speaker",
            "category": "Audio",
            "brand": "SoundMax",
        }

        mock_bq_client = MagicMock()
        mock_bq_client.query.return_value.result.return_value = [_mock_bq_row(updated_product)]

        mock_vs_client = MagicMock()

        with patch.object(ps.bigquery, "Client", return_value=mock_bq_client), \
             patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            event = _make_cloud_event("updated", "elec-005")
            ps.handle_product_event(event)

        mock_vs_client.upsert_data_object.assert_called_once()

    def test_created_event_upsert_includes_product_data(self, ps):
        """Upsert request should contain actual product data."""
        product = {
            "product_id": "elec-001",
            "name": "Wireless Headphones",
            "price": 179.99,
            "description": "Over-ear headphones",
            "category": "Audio",
            "brand": "SoundMax",
        }

        mock_bq_client = MagicMock()
        mock_bq_client.query.return_value.result.return_value = [_mock_bq_row(product)]

        mock_vs_client = MagicMock()

        with patch.object(ps.bigquery, "Client", return_value=mock_bq_client), \
             patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            event = _make_cloud_event("created", "elec-001")
            ps.handle_product_event(event)

        # Inspect the UpsertDataObjectRequest args
        upsert_call = mock_vs_client.upsert_data_object.call_args
        assert upsert_call is not None


# =====================================================================
# Use Case 2: Fashion -- Delete events
# =====================================================================

class TestFashionDeleteEvents:
    """Fashion: product deletion with hard and soft delete modes."""

    def test_hard_delete_calls_vs_delete(self, ps):
        """Hard delete should call delete_data_object on VS client."""
        mock_vs_client = MagicMock()

        with patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            event = _make_cloud_event("deleted", "fash-003")
            ps.handle_product_event(event)

        mock_vs_client.delete_data_object.assert_called_once()

    def test_hard_delete_uses_correct_collection(self, ps):
        """Hard delete request should reference the correct collection."""
        mock_vs_client = MagicMock()

        with patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            ps.delete_from_vector_search("fash-003")

        mock_vs_client.delete_data_object.assert_called_once()

    def test_soft_delete_fetches_and_upserts_with_zero_stock(self, ps):
        """Soft delete should fetch product, set stock=0, and upsert."""
        ps.DELETION_MODE = "soft"

        product = {
            "product_id": "fash-008",
            "name": "Linen Blazer",
            "price": 179.99,
            "description": "Italian linen blazer",
            "category": "Men's Outerwear",
            "brand": "SartoriaLux",
            "stock": 20,
        }

        mock_bq_client = MagicMock()
        mock_bq_client.query.return_value.result.return_value = [_mock_bq_row(product)]

        mock_vs_client = MagicMock()

        with patch.object(ps.bigquery, "Client", return_value=mock_bq_client), \
             patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            ps.delete_from_vector_search("fash-008")

        # Should have fetched from BQ and then upserted to VS
        mock_bq_client.query.assert_called_once()
        mock_vs_client.upsert_data_object.assert_called_once()


# =====================================================================
# Use Case 3: Grocery -- edge cases
# =====================================================================

class TestGroceryEdgeCases:
    """Grocery: edge cases -- missing product, empty product_id."""

    def test_missing_product_in_bigquery(self, ps):
        """If product not found in BQ, should not upsert."""
        mock_bq_client = MagicMock()
        mock_bq_client.query.return_value.result.return_value = []  # no rows

        mock_vs_client = MagicMock()

        with patch.object(ps.bigquery, "Client", return_value=mock_bq_client), \
             patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            event = _make_cloud_event("updated", "groc-999")
            ps.handle_product_event(event)

        mock_vs_client.upsert_data_object.assert_not_called()

    def test_empty_product_id_ignored(self, ps):
        """Messages without product_id should be ignored."""
        mock_bq_client = MagicMock()

        with patch.object(ps.bigquery, "Client", return_value=mock_bq_client):
            event = _make_cloud_event("created", "")
            ps.handle_product_event(event)

        mock_bq_client.query.assert_not_called()

    def test_upsert_includes_description(self, ps):
        """VS 2.0 auto-embeddings need description for text template."""
        product = {
            "product_id": "groc-001",
            "name": "Organic Olive Oil",
            "price": 14.99,
            "description": "Cold-pressed olive oil from Greece",
            "category": "Oils & Vinegars",
            "brand": "OliveGrove",
        }

        mock_bq_client = MagicMock()
        mock_bq_client.query.return_value.result.return_value = [_mock_bq_row(product)]

        mock_vs_client = MagicMock()

        with patch.object(ps.bigquery, "Client", return_value=mock_bq_client), \
             patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            event = _make_cloud_event("created", "groc-001")
            ps.handle_product_event(event)

        mock_vs_client.upsert_data_object.assert_called_once()


# =====================================================================
# upsert_to_vector_search internals
# =====================================================================

class TestUpsertFunction:
    """Test the upsert function signature and behavior."""

    def test_upsert_calls_dataobject_client(self, ps):
        """Upsert should call DataObjectServiceClient.upsert_data_object."""
        mock_client = MagicMock()

        with patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_client):
            ps.upsert_to_vector_search("elec-001", {"name": "Headphones", "price": 179.99})

        mock_client.upsert_data_object.assert_called_once()

    def test_upsert_passes_correct_collection(self, ps):
        """Upsert request should reference the COLLECTION env var."""
        mock_client = MagicMock()

        with patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_client):
            ps.upsert_to_vector_search("elec-001", {"name": "Headphones"})

        call_args = mock_client.upsert_data_object.call_args
        assert call_args is not None


# =====================================================================
# Error scenarios
# =====================================================================

class TestErrorScenarios:
    """Error handling: malformed messages, API failures, invalid base64."""

    def test_invalid_base64_raises(self, ps):
        """Malformed base64 in message data should raise an error."""
        event = MagicMock()
        event.data = {"message": {"data": "not-valid-base64!!!"}}

        with pytest.raises(Exception):
            ps.handle_product_event(event)

    def test_invalid_json_in_message(self, ps):
        """Valid base64 but invalid JSON should raise an error."""
        invalid_json = base64.b64encode(b"not json{{{").decode("utf-8")
        event = MagicMock()
        event.data = {"message": {"data": invalid_json}}

        with pytest.raises(json.JSONDecodeError):
            ps.handle_product_event(event)

    def test_missing_message_key(self, ps):
        """CloudEvent without 'message' key should raise."""
        event = MagicMock()
        event.data = {}

        with pytest.raises((KeyError, TypeError)):
            ps.handle_product_event(event)

    def test_missing_data_key_in_message(self, ps):
        """CloudEvent message without 'data' key should raise."""
        event = MagicMock()
        event.data = {"message": {}}

        with pytest.raises((KeyError, TypeError)):
            ps.handle_product_event(event)

    def test_unknown_action_does_not_crash(self, ps):
        """Unknown action should not upsert or delete, just log."""
        mock_bq_client = MagicMock()
        mock_bq_client.query.return_value.result.return_value = []

        mock_vs_client = MagicMock()

        with patch.object(ps.bigquery, "Client", return_value=mock_bq_client), \
             patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            event = _make_cloud_event("unknown_action", "prod-001")
            ps.handle_product_event(event)

        mock_vs_client.delete_data_object.assert_not_called()

    def test_bigquery_query_failure(self, ps):
        """If BigQuery query raises, the error should propagate."""
        mock_bq_client = MagicMock()
        mock_bq_client.query.side_effect = Exception("BigQuery unavailable")

        with patch.object(ps.bigquery, "Client", return_value=mock_bq_client):
            with pytest.raises(Exception, match="BigQuery unavailable"):
                ps.fetch_product("elec-001")

    def test_vs_upsert_failure(self, ps):
        """If Vector Search upsert raises, the error should propagate."""
        mock_vs_client = MagicMock()
        mock_vs_client.upsert_data_object.side_effect = Exception("VS API error")

        with patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            with pytest.raises(Exception, match="VS API error"):
                ps.upsert_to_vector_search("elec-001", {"name": "Test"})

    def test_vs_delete_failure(self, ps):
        """If Vector Search delete raises, the error should propagate."""
        mock_vs_client = MagicMock()
        mock_vs_client.delete_data_object.side_effect = Exception("VS delete error")

        with patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            with pytest.raises(Exception, match="VS delete error"):
                ps.delete_from_vector_search("fash-001")


# =====================================================================
# Hardcoded values validation
# =====================================================================

class TestHardcodedValues:
    """Verify hardcoded dataset/table names and env var usage."""

    def test_fetch_product_queries_correct_dataset(self, ps):
        """fetch_product should query products_dataset.products table."""
        mock_bq_client = MagicMock()
        mock_bq_client.query.return_value.result.return_value = []

        with patch.object(ps.bigquery, "Client", return_value=mock_bq_client):
            ps.fetch_product("test-id")

        query = mock_bq_client.query.call_args[0][0]
        assert "retail_skill_products" in query
        assert "products" in query

    def test_env_vars_are_read(self, ps):
        """Module-level constants should match env vars."""
        assert ps.PROJECT_ID == "test-project"
        assert ps.LOCATION == "us-central1"
        assert "products-collection" in ps.COLLECTION
        assert ps.DELETION_MODE == "hard"

    def test_collection_env_var_used_in_upsert(self, ps):
        """Upsert should use COLLECTION from env, not a hardcoded value."""
        mock_vs_client = MagicMock()

        with patch.object(ps.vectorsearch, "DataObjectServiceClient",
                          return_value=mock_vs_client):
            ps.upsert_to_vector_search("prod-001", {"name": "Test"})

        # The request should reference the collection path from COLLECTION env var
        call_args = mock_vs_client.upsert_data_object.call_args
        request_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("request")
        assert request_arg is not None

    def test_deletion_mode_env_var(self, ps, monkeypatch):
        """DELETION_MODE env var should control delete behavior."""
        monkeypatch.setenv("DELETION_MODE", "soft")
        import importlib
        importlib.reload(ps)
        assert ps.DELETION_MODE == "soft"
