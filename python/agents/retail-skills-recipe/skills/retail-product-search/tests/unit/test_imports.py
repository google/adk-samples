"""Import validation tests for retail-product-search scripts.

Verifies that all scripts can be imported without errors and that
key functions/constants are accessible after import.
"""

import pytest


class TestScriptImports:
    """Verify all scripts import successfully with mocked GCP modules."""

    def test_import_validate_schema(self):
        import validate_schema
        assert hasattr(validate_schema, "validate")
        assert hasattr(validate_schema, "load_records")
        assert hasattr(validate_schema, "FIELD_LEVELS")

    def test_import_ingest_bigquery(self):
        import ingest_bigquery
        assert hasattr(ingest_bigquery, "validate_product")
        assert hasattr(ingest_bigquery, "convert_types")
        assert hasattr(ingest_bigquery, "load_from_csv")
        assert hasattr(ingest_bigquery, "ingest")
        assert hasattr(ingest_bigquery, "REQUIRED_FIELDS")
        assert hasattr(ingest_bigquery, "SCHEMA")

    def test_import_ingest_vertex_search(self):
        import ingest_vertex_search
        assert hasattr(ingest_vertex_search, "build_text_template")
        assert hasattr(ingest_vertex_search, "get_data_schema")
        assert hasattr(ingest_vertex_search, "create_collection_if_needed")
        assert hasattr(ingest_vertex_search, "ingest_products")
        assert hasattr(ingest_vertex_search, "PRODUCT_DATA_FIELDS")
        assert hasattr(ingest_vertex_search, "BATCH_SIZE")

    def test_import_pubsub_sync(self):
        import pubsub_sync
        assert hasattr(pubsub_sync, "handle_product_event")
        assert hasattr(pubsub_sync, "fetch_product")
        assert hasattr(pubsub_sync, "upsert_to_vector_search")
        assert hasattr(pubsub_sync, "delete_from_vector_search")

    def test_import_cleanup(self):
        import cleanup
        assert hasattr(cleanup, "cleanup")
        assert hasattr(cleanup, "delete_bigquery")
        assert hasattr(cleanup, "delete_gcs")
        assert hasattr(cleanup, "ALL_RESOURCE_TYPES")

    def test_import_ingest_gcs(self):
        import ingest_gcs
        assert hasattr(ingest_gcs, "main") or callable(getattr(ingest_gcs, "upload_images", None))

    def test_import_api_connector(self):
        import api_connector
        assert hasattr(api_connector, "main") or hasattr(api_connector, "search")


class TestModuleConstants:
    """Verify key constants are defined correctly after import."""

    def test_validate_schema_field_levels_has_all_levels(self):
        from validate_schema import FIELD_LEVELS
        assert "Basic" in FIELD_LEVELS
        assert "Standard" in FIELD_LEVELS
        assert "Extended" in FIELD_LEVELS
        assert "Full" in FIELD_LEVELS

    def test_ingest_bigquery_required_fields(self):
        from ingest_bigquery import REQUIRED_FIELDS
        assert "product_id" in REQUIRED_FIELDS
        assert "name" in REQUIRED_FIELDS
        assert "price" in REQUIRED_FIELDS
        assert "description" in REQUIRED_FIELDS

    def test_ingest_vertex_search_product_data_fields(self):
        from ingest_vertex_search import PRODUCT_DATA_FIELDS
        assert PRODUCT_DATA_FIELDS["price"] == "number"
        assert PRODUCT_DATA_FIELDS["stock"] == "integer"
        assert PRODUCT_DATA_FIELDS["product_id"] == "string"

    def test_cleanup_all_resource_types(self):
        from cleanup import ALL_RESOURCE_TYPES
        assert len(ALL_RESOURCE_TYPES) >= 6
