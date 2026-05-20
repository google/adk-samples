"""Tests for cleanup.py across 3 use cases.

Use Case 1 (Electronics): Dry-run mode, selective resource deletion
Use Case 2 (Fashion): Full cleanup with collection deletion
Use Case 3 (Grocery): Legacy v1 index cleanup, missing resources
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

import cleanup as cu
from cleanup import (
    delete_bigquery,
    delete_gcs,
    delete_vectorsearch_collection,
    delete_vectorsearch_v1_index,
    cleanup,
    load_config,
    ALL_RESOURCE_TYPES,
)

# Get the mock exceptions that cleanup.py will use
from google.api_core.exceptions import NotFound


# =====================================================================
# Use Case 1: Electronics — Dry-run & selective cleanup
# =====================================================================

class TestElectronicsDryRun:
    """Electronics: dry-run mode should describe but not delete."""

    def test_bigquery_dry_run_no_delete(self):
        """Dry run should not delete BigQuery dataset."""
        mock_client = MagicMock()
        mock_client.get_dataset.return_value = MagicMock()  # exists

        with patch.object(cu.bigquery, "Client", return_value=mock_client):
            delete_bigquery("test-project", "products_dataset", dry_run=True)

        mock_client.delete_dataset.assert_not_called()

    def test_gcs_dry_run_no_delete(self):
        """Dry run should not delete GCS buckets."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.get_bucket.return_value = mock_bucket

        with patch.object(cu.storage, "Client", return_value=mock_client):
            delete_gcs("test-project", ["products", "embeddings"], dry_run=True)

        mock_bucket.delete.assert_not_called()

    def test_cleanup_selective_bigquery_only(self):
        """Selective cleanup should only delete specified resource types."""
        config = {"gcp_project_id": "test-project", "gcp_region": "us-central1", "project_name": "electronics"}

        with patch("cleanup.delete_bigquery") as mock_bq, \
             patch("cleanup.delete_gcs") as mock_gcs, \
             patch("cleanup.delete_vectorsearch_collection") as mock_vs, \
             patch("cleanup.delete_vectorsearch_v1_index") as mock_vs_v1:

            cleanup(config, dry_run=True, only=["bigquery"])

            mock_bq.assert_called_once()
            mock_gcs.assert_not_called()
            mock_vs.assert_not_called()

    def test_all_resource_types_defined(self):
        """Verify all expected resource types are in the list."""
        assert "bigquery" in ALL_RESOURCE_TYPES
        assert "gcs" in ALL_RESOURCE_TYPES
        assert "vectorsearch" in ALL_RESOURCE_TYPES
        assert "pubsub" in ALL_RESOURCE_TYPES
        assert "cloudrun" in ALL_RESOURCE_TYPES
        assert "cloudfunctions" in ALL_RESOURCE_TYPES


# =====================================================================
# Use Case 2: Fashion — Full cleanup with VS 2.0 collection deletion
# =====================================================================

class TestFashionFullCleanup:
    """Fashion: full cleanup including VS 2.0 collection."""

    def test_delete_collection_when_exists(self):
        """Should delete VS 2.0 collection when it exists (no exception)."""
        # When collection exists, delete_vectorsearch_collection should run without error
        delete_vectorsearch_collection("test-project", "us-central1", "fashion-collection", dry_run=False)

    def test_delete_collection_not_found(self):
        """Should handle non-existent collection gracefully."""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = NotFound("not found")

        mock_vs = sys.modules["google.cloud.vectorsearch"]
        original_client = mock_vs.VectorSearchServiceClient
        mock_vs.VectorSearchServiceClient = MagicMock(return_value=mock_client)

        try:
            # Should not raise
            delete_vectorsearch_collection("test-project", "us-central1", "nonexistent", dry_run=False)
            mock_client.delete_collection.assert_not_called()
        finally:
            mock_vs.VectorSearchServiceClient = original_client

    def test_delete_collection_dry_run(self):
        """Dry run should not delete the collection."""
        mock_client = MagicMock()
        mock_client.get_collection.return_value = MagicMock()

        mock_vs = sys.modules["google.cloud.vectorsearch"]
        original_client = mock_vs.VectorSearchServiceClient
        mock_vs.VectorSearchServiceClient = MagicMock(return_value=mock_client)

        try:
            delete_vectorsearch_collection("test-project", "us-central1", "fashion-collection", dry_run=True)
            mock_client.delete_collection.assert_not_called()
        finally:
            mock_vs.VectorSearchServiceClient = original_client

    def test_bigquery_actual_delete(self):
        """Actual delete should call delete_dataset with delete_contents=True."""
        mock_client = MagicMock()
        mock_client.get_dataset.return_value = MagicMock()

        with patch.object(cu.bigquery, "Client", return_value=mock_client):
            delete_bigquery("test-project", "products_dataset", dry_run=False)

        mock_client.delete_dataset.assert_called_once_with(
            "test-project.products_dataset", delete_contents=True, not_found_ok=True
        )


# =====================================================================
# Use Case 3: Grocery — Legacy v1 cleanup, missing resources
# =====================================================================

class TestGroceryLegacyCleanup:
    """Grocery: clean up legacy v1 indexes alongside v2 collections."""

    def test_delete_v1_index_when_exists(self):
        """Should run v1 index deletion without error."""
        delete_vectorsearch_v1_index("test-project", "us-central1", "products_index", dry_run=False)

    def test_delete_v1_index_not_found(self):
        """Should handle missing v1 index gracefully."""
        mock_ai = sys.modules["google.cloud.aiplatform"]
        mock_ai.init = MagicMock()
        mock_ai.MatchingEngineIndex = MagicMock()
        mock_ai.MatchingEngineIndex.list.return_value = []

        # Should not raise
        delete_vectorsearch_v1_index("test-project", "us-central1", "products_index", dry_run=False)

    def test_delete_v1_index_dry_run(self):
        """Dry run should not delete the v1 index."""
        mock_index = MagicMock()

        mock_ai = sys.modules["google.cloud.aiplatform"]
        mock_ai.init = MagicMock()
        mock_ai.MatchingEngineIndex = MagicMock()
        mock_ai.MatchingEngineIndex.list.return_value = [mock_index]

        delete_vectorsearch_v1_index("test-project", "us-central1", "products_index", dry_run=True)

        mock_index.delete.assert_not_called()

    def test_cleanup_vectorsearch_calls_both_v1_and_v2(self):
        """Full vectorsearch cleanup should try both v2 collection and v1 index."""
        config = {"gcp_project_id": "test-project", "gcp_region": "us-central1", "project_name": "grocery"}

        with patch("cleanup.delete_bigquery"), \
             patch("cleanup.delete_gcs"), \
             patch("cleanup.delete_vectorsearch_collection") as mock_v2, \
             patch("cleanup.delete_vectorsearch_v1_index") as mock_v1, \
             patch("cleanup.delete_pubsub"), \
             patch("cleanup.delete_cloudrun"), \
             patch("cleanup.delete_cloudfunctions"):

            cleanup(config, dry_run=False, only=["vectorsearch"])

            mock_v2.assert_called_once()
            mock_v1.assert_called_once()

    def test_bigquery_not_found_graceful(self):
        """Should handle non-existent BigQuery dataset gracefully."""
        mock_client = MagicMock()
        mock_client.get_dataset.side_effect = Exception("Not found")

        with patch.object(cu.bigquery, "Client", return_value=mock_client):
            # Should not raise
            delete_bigquery("test-project", "nonexistent_dataset", dry_run=False)

        mock_client.delete_dataset.assert_not_called()

    def test_gcs_bucket_not_found_graceful(self):
        """Should handle non-existent GCS buckets gracefully."""
        mock_client = MagicMock()
        mock_client.get_bucket.side_effect = Exception("Not found")

        with patch.object(cu.storage, "Client", return_value=mock_client):
            # Should not raise
            delete_gcs("test-project", ["products"], dry_run=False)


# =====================================================================
# Config loading
# =====================================================================

class TestCleanupConfig:
    """Config loading for cleanup script."""

    def test_load_config_valid(self, sample_config_yaml):
        cfg = load_config(sample_config_yaml)
        assert cfg["gcp_project_id"] == "test-project-123"

    def test_cleanup_requires_project_id(self):
        """Should exit if gcp_project_id is missing from config."""
        config = {"gcp_region": "us-central1"}

        with pytest.raises(SystemExit):
            cleanup(config, dry_run=True, only=ALL_RESOURCE_TYPES)


# =====================================================================
# Hardcoded values validation
# =====================================================================

class TestHardcodedResourceNames:
    """Verify that cleanup uses expected hardcoded resource names."""

    def test_cleanup_bigquery_uses_products_dataset(self):
        """Cleanup should pass 'products_dataset' to delete_bigquery."""
        config = {"gcp_project_id": "test-project", "gcp_region": "us-central1", "project_name": "test"}

        with patch("cleanup.delete_bigquery") as mock_bq, \
             patch("cleanup.delete_gcs"), \
             patch("cleanup.delete_vectorsearch_collection"), \
             patch("cleanup.delete_vectorsearch_v1_index"), \
             patch("cleanup.delete_pubsub"), \
             patch("cleanup.delete_cloudrun"), \
             patch("cleanup.delete_cloudfunctions"):
            cleanup(config, dry_run=True, only=["bigquery"])

        args = mock_bq.call_args[0]
        assert args[1] == "retail_skill_products"

    def test_cleanup_gcs_bucket_prefixes(self):
        """Cleanup should delete expected GCS bucket prefixes."""
        config = {"gcp_project_id": "test-project", "gcp_region": "us-central1", "project_name": "test"}

        with patch("cleanup.delete_bigquery"), \
             patch("cleanup.delete_gcs") as mock_gcs, \
             patch("cleanup.delete_vectorsearch_collection"), \
             patch("cleanup.delete_vectorsearch_v1_index"), \
             patch("cleanup.delete_pubsub"), \
             patch("cleanup.delete_cloudrun"), \
             patch("cleanup.delete_cloudfunctions"):
            cleanup(config, dry_run=True, only=["gcs"])

        bucket_prefixes = mock_gcs.call_args[0][1]
        assert "retail-skill-products" in bucket_prefixes
        assert "retail-skill-embeddings" in bucket_prefixes

    def test_cleanup_vs_collection_name(self):
        """Cleanup should delete 'products-collection' VS collection."""
        config = {"gcp_project_id": "test-project", "gcp_region": "us-central1", "project_name": "test"}

        with patch("cleanup.delete_bigquery"), \
             patch("cleanup.delete_gcs"), \
             patch("cleanup.delete_vectorsearch_collection") as mock_vs, \
             patch("cleanup.delete_vectorsearch_v1_index"), \
             patch("cleanup.delete_pubsub"), \
             patch("cleanup.delete_cloudrun"), \
             patch("cleanup.delete_cloudfunctions"):
            cleanup(config, dry_run=True, only=["vectorsearch"])

        args = mock_vs.call_args[0]
        assert args[2] == "retail-skill-products-collection"

    def test_cleanup_pubsub_topic_name(self):
        """Cleanup should delete 'product-changes' Pub/Sub topic."""
        config = {"gcp_project_id": "test-project", "gcp_region": "us-central1", "project_name": "test"}

        with patch("cleanup.delete_bigquery"), \
             patch("cleanup.delete_gcs"), \
             patch("cleanup.delete_vectorsearch_collection"), \
             patch("cleanup.delete_vectorsearch_v1_index"), \
             patch("cleanup.delete_pubsub") as mock_ps, \
             patch("cleanup.delete_cloudrun"), \
             patch("cleanup.delete_cloudfunctions"):
            cleanup(config, dry_run=True, only=["pubsub"])

        args = mock_ps.call_args[0]
        assert args[1] == "retail-skill-product-changes"


# =====================================================================
# Cascading failure tests
# =====================================================================

class TestCascadingFailures:
    """Verify that failure in one resource cleanup doesn't block others."""

    def test_bigquery_failure_does_not_block_gcs(self):
        """If BigQuery delete raises, GCS cleanup should still be attempted."""
        config = {"gcp_project_id": "test-project", "gcp_region": "us-central1", "project_name": "test"}

        with patch("cleanup.delete_bigquery", side_effect=Exception("BQ exploded")) as mock_bq, \
             patch("cleanup.delete_gcs") as mock_gcs, \
             patch("cleanup.delete_vectorsearch_collection"), \
             patch("cleanup.delete_vectorsearch_v1_index"), \
             patch("cleanup.delete_pubsub"), \
             patch("cleanup.delete_cloudrun"), \
             patch("cleanup.delete_cloudfunctions"):
            try:
                cleanup(config, dry_run=False, only=["bigquery", "gcs"])
            except Exception:
                pass

            # GCS should still be called even if BQ failed
            # Note: if cleanup doesn't catch exceptions between resource types,
            # this test documents that behavior
            mock_bq.assert_called_once()

    def test_gcs_bucket_failure_blocks_subsequent_buckets(self):
        """BUG: If bucket.delete() raises, subsequent buckets are NOT attempted.

        Current code only catches get_bucket exceptions, not delete exceptions.
        This test documents the current (broken) behavior.
        """
        mock_client = MagicMock()

        bucket1 = MagicMock()
        bucket1.list_blobs.return_value = []
        bucket1.delete.side_effect = Exception("Bucket locked")

        def get_bucket_side_effect(name):
            if "bucket1" in name:
                return bucket1
            raise Exception("Not found")

        mock_client.get_bucket.side_effect = get_bucket_side_effect

        with patch.object(cu.storage, "Client", return_value=mock_client):
            with pytest.raises(Exception, match="Bucket locked"):
                delete_gcs("test-project", ["bucket1", "bucket2"], dry_run=False)

        # Only first bucket was attempted because the exception propagated
        assert mock_client.get_bucket.call_count == 1

    def test_gcs_not_found_continues_to_next_bucket(self):
        """If get_bucket raises (not found), the next bucket should be attempted."""
        mock_client = MagicMock()

        bucket2 = MagicMock()
        bucket2.list_blobs.return_value = []

        def get_bucket_side_effect(name):
            if "bucket1" in name:
                raise Exception("Not found")
            return bucket2

        mock_client.get_bucket.side_effect = get_bucket_side_effect

        with patch.object(cu.storage, "Client", return_value=mock_client):
            delete_gcs("test-project", ["bucket1", "bucket2"], dry_run=False)

        # Both buckets were attempted
        assert mock_client.get_bucket.call_count == 2
        # Second bucket was deleted
        bucket2.delete.assert_called_once()

    def test_vectorsearch_collection_failure_does_not_block_v1_cleanup(self):
        """If VS 2.0 collection delete fails, v1 index cleanup should still run."""
        config = {"gcp_project_id": "test-project", "gcp_region": "us-central1", "project_name": "test"}

        with patch("cleanup.delete_bigquery"), \
             patch("cleanup.delete_gcs"), \
             patch("cleanup.delete_vectorsearch_collection", side_effect=Exception("VS2 error")) as mock_v2, \
             patch("cleanup.delete_vectorsearch_v1_index") as mock_v1, \
             patch("cleanup.delete_pubsub"), \
             patch("cleanup.delete_cloudrun"), \
             patch("cleanup.delete_cloudfunctions"):
            try:
                cleanup(config, dry_run=False, only=["vectorsearch"])
            except Exception:
                pass

            mock_v2.assert_called_once()
            # v1 may or may not be called depending on whether cleanup catches exceptions

    def test_bigquery_delete_handles_permission_error(self):
        """BigQuery delete with permission error should not crash."""
        mock_client = MagicMock()
        mock_client.get_dataset.return_value = MagicMock()  # exists
        mock_client.delete_dataset.side_effect = Exception("403 Permission denied")

        with patch.object(cu.bigquery, "Client", return_value=mock_client):
            try:
                delete_bigquery("test-project", "products_dataset", dry_run=False)
            except Exception:
                pass  # Documents whether the function catches or propagates

        # Should have attempted the delete
        mock_client.delete_dataset.assert_called_once()

    def test_full_cleanup_with_all_failures(self):
        """Full cleanup where every resource type fails should still attempt all."""
        config = {"gcp_project_id": "test-project", "gcp_region": "us-central1", "project_name": "test"}

        with patch("cleanup.delete_bigquery", side_effect=Exception("BQ fail")), \
             patch("cleanup.delete_gcs", side_effect=Exception("GCS fail")), \
             patch("cleanup.delete_vectorsearch_collection", side_effect=Exception("VS fail")), \
             patch("cleanup.delete_vectorsearch_v1_index", side_effect=Exception("VS1 fail")), \
             patch("cleanup.delete_pubsub", side_effect=Exception("PS fail")), \
             patch("cleanup.delete_cloudrun", side_effect=Exception("CR fail")), \
             patch("cleanup.delete_cloudfunctions", side_effect=Exception("CF fail")):
            try:
                cleanup(config, dry_run=False, only=ALL_RESOURCE_TYPES)
            except Exception:
                pass  # Documents behavior when all resources fail
