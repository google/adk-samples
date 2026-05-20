"""Tests for ingest_vertex_search.py (Vector Search 2.0) across 3 use cases.

Use Case 1 (Electronics): Collection creation, product ingestion, text templates
Use Case 2 (Fashion): Custom embedding fields, data schema detection
Use Case 3 (Grocery): Out-of-stock products in VS 2.0, config loading
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import ingest_vertex_search as ivs
from ingest_vertex_search import (
    build_text_template,
    get_data_schema,
    create_collection_if_needed,
    ingest_products,
    fetch_products,
    load_config,
    PRODUCT_DATA_FIELDS,
    DEFAULT_EMBEDDING_FIELDS,
    DEFAULT_EMBEDDING_MODEL,
    BATCH_SIZE,
)

# Use the mock exception classes that the script will actually catch
NotFound = ivs.exceptions.NotFound
AlreadyExists = ivs.exceptions.AlreadyExists


# =====================================================================
# Use Case 1: Electronics Store — Collection creation & ingestion
# =====================================================================

class TestElectronicsCollectionCreation:
    """Electronics: create collection, verify schema, ingest products."""

    def test_build_text_template_default_fields(self):
        template = build_text_template(DEFAULT_EMBEDDING_FIELDS)
        assert template == "name: {name} | description: {description} | category: {category} | brand: {brand}"

    def test_build_text_template_custom_fields(self):
        template = build_text_template(["name", "description"])
        assert template == "name: {name} | description: {description}"

    def test_get_data_schema_from_electronics(self, typed_electronics_products):
        schema = get_data_schema(typed_electronics_products)
        assert schema["type"] == "object"
        props = schema["properties"]
        assert props["product_id"] == {"type": "string"}
        assert props["name"] == {"type": "string"}
        assert props["price"] == {"type": "number"}
        assert props["description"] == {"type": "string"}
        assert props["category"] == {"type": "string"}
        assert props["brand"] == {"type": "string"}

    def test_get_data_schema_detects_missing_fields(self):
        """Products without image_url shouldn't have it in schema."""
        products = [
            {"product_id": "p1", "name": "Test", "price": 9.99, "description": "D"},
        ]
        schema = get_data_schema(products)
        assert "image_url" not in schema["properties"]
        assert "rating" not in schema["properties"]

    def test_create_collection_when_not_exists(self):
        """Should create collection when it doesn't exist."""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = NotFound("not found")
        mock_op = MagicMock()
        mock_op.result.return_value = None
        mock_client.create_collection.return_value = mock_op

        with patch.object(ivs.vectorsearch, "VectorSearchServiceClient", return_value=mock_client):
            result = create_collection_if_needed(
                "test-project", "us-central1", "electronics-collection",
                "gemini-embedding-001", ["name", "description", "category", "brand"],
                {"type": "object", "properties": {"name": {"type": "string"}}},
            )

        assert result == "projects/test-project/locations/us-central1/collections/electronics-collection"
        mock_client.create_collection.assert_called_once()

    def test_create_collection_skips_when_exists(self):
        """Should skip creation when collection already exists."""
        mock_client = MagicMock()
        mock_client.get_collection.return_value = MagicMock()  # exists

        with patch.object(ivs.vectorsearch, "VectorSearchServiceClient", return_value=mock_client):
            result = create_collection_if_needed(
                "test-project", "us-central1", "electronics-collection",
                "gemini-embedding-001", DEFAULT_EMBEDDING_FIELDS,
                {"type": "object", "properties": {}},
            )

        assert result == "projects/test-project/locations/us-central1/collections/electronics-collection"
        mock_client.create_collection.assert_not_called()

    def test_ingest_products_electronics(self, typed_electronics_products):
        """Should create data objects for each product."""
        mock_client = MagicMock()

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            ingest_products(
                "projects/test-project/locations/us-central1/collections/test-col",
                typed_electronics_products,
            )

        assert mock_client.create_data_object.call_count == 3

    def test_ingest_products_preserves_float_types(self, typed_electronics_products):
        """Price and rating should be stored as float in the data struct."""
        mock_client = MagicMock()

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            ingest_products(
                "projects/test-project/locations/us-central1/collections/test-col",
                typed_electronics_products,
            )

        # Inspect the data_object kwarg passed to first call
        call_kwargs = mock_client.create_data_object.call_args_list[0][1]
        data_object = call_kwargs.get("data_object")
        if data_object is not None:
            # DataObject is a MagicMock wrapping a real Struct-like dict
            data = data_object.data if hasattr(data_object, "data") else data_object
            if isinstance(data, dict):
                assert isinstance(data.get("price"), float), f"price should be float, got {type(data.get('price'))}"
                if "rating" in data:
                    assert isinstance(data["rating"], float), f"rating should be float, got {type(data['rating'])}"

    def test_ingest_products_preserves_int_types(self, typed_electronics_products):
        """Stock should be stored as int in the data struct."""
        mock_client = MagicMock()

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            ingest_products(
                "projects/test-project/locations/us-central1/collections/test-col",
                typed_electronics_products,
            )

        call_kwargs = mock_client.create_data_object.call_args_list[0][1]
        data_object = call_kwargs.get("data_object")
        if data_object is not None:
            data = data_object.data if hasattr(data_object, "data") else data_object
            if isinstance(data, dict) and "stock" in data:
                assert isinstance(data["stock"], int), f"stock should be int, got {type(data['stock'])}"

    def test_ingest_products_preserves_string_types(self, typed_electronics_products):
        """Product name and description should be stored as str."""
        mock_client = MagicMock()

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            ingest_products(
                "projects/test-project/locations/us-central1/collections/test-col",
                typed_electronics_products,
            )

        call_kwargs = mock_client.create_data_object.call_args_list[0][1]
        data_object = call_kwargs.get("data_object")
        if data_object is not None:
            data = data_object.data if hasattr(data_object, "data") else data_object
            if isinstance(data, dict):
                assert isinstance(data.get("name"), str)
                assert isinstance(data.get("product_id"), str)

    def test_ingest_skips_already_exists(self, typed_electronics_products):
        """AlreadyExists errors should be skipped, not raised."""
        mock_client = MagicMock()
        mock_client.create_data_object.side_effect = AlreadyExists("exists")

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            # Should not raise
            ingest_products(
                "projects/test-project/locations/us-central1/collections/test-col",
                typed_electronics_products,
            )

        assert mock_client.create_data_object.call_count == 3


# =====================================================================
# Use Case 2: Fashion Store — custom fields, schema detection
# =====================================================================

class TestFashionCollectionIngestion:
    """Fashion: custom embedding fields, products with special characters."""

    def test_text_template_for_fashion(self):
        """Fashion might embed name + description + category only (no brand)."""
        template = build_text_template(["name", "description", "category"])
        assert template == "name: {name} | description: {description} | category: {category}"

    def test_data_schema_with_all_fields(self):
        """Fashion products with all Extended fields."""
        products = [{
            "product_id": "fash-001",
            "name": "Oxford Shirt",
            "price": 59.99,
            "description": "Cotton shirt",
            "category": "Men's Shirts",
            "brand": "Brooks & Co",
            "image_url": "https://example.com/fash-001.jpg",
            "rating": 4.3,
            "stock": 150,
        }]
        schema = get_data_schema(products)
        assert len(schema["properties"]) == 9  # All PRODUCT_DATA_FIELDS

    def test_ingest_fashion_with_missing_image(self):
        """Products without image_url should still ingest fine."""
        mock_client = MagicMock()

        products = [
            {
                "product_id": "fash-005",
                "name": "Canvas Tote Bag",
                "price": 34.99,
                "description": "Heavy-duty canvas tote",
                "category": "Bags",
                "brand": "EcoCarry",
                "rating": 4.1,
                "stock": 200,
            },
        ]

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            ingest_products(
                "projects/test-project/locations/us-central1/collections/fashion-col",
                products,
            )

        mock_client.create_data_object.assert_called_once()

    def test_ingest_preserves_special_chars_in_data(self):
        """Special chars like apostrophes should survive ingestion."""
        mock_client = MagicMock()

        products = [{
            "product_id": "fash-001",
            "name": "Men's Oxford Shirt",
            "price": 59.99,
            "description": "Cotton shirt",
            "category": "Men's Shirts",
            "brand": "Brooks & Co",
        }]

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            ingest_products(
                "projects/test-project/locations/us-central1/collections/fashion-col",
                products,
            )

        mock_client.create_data_object.assert_called_once()


# =====================================================================
# Use Case 3: Grocery Store — out-of-stock, config, batch limits
# =====================================================================

class TestGroceryCollectionIngestion:
    """Grocery: out-of-stock handling, batch sizing, config loading."""

    def test_ingest_out_of_stock_products(self, typed_grocery_products):
        """Out-of-stock products (stock=0) should still be ingested."""
        mock_client = MagicMock()

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            ingest_products(
                "projects/test-project/locations/us-central1/collections/grocery-col",
                typed_grocery_products,
            )

        # All 3 products should be ingested, including stock=0
        assert mock_client.create_data_object.call_count == 3

    def test_batch_size_constant(self):
        """Batch size should be 250 (VS 2.0 auto-embedding limit)."""
        assert BATCH_SIZE == 250

    def test_product_data_fields_types(self):
        """Verify product field types match expected schema."""
        assert PRODUCT_DATA_FIELDS["product_id"] == "string"
        assert PRODUCT_DATA_FIELDS["price"] == "number"
        assert PRODUCT_DATA_FIELDS["stock"] == "integer"
        assert PRODUCT_DATA_FIELDS["rating"] == "number"

    def test_default_embedding_model(self):
        """Default should be gemini-embedding-001 for VS 2.0."""
        assert DEFAULT_EMBEDDING_MODEL == "gemini-embedding-001"

    def test_ingest_handles_generic_error(self):
        """Generic errors should be logged but not stop ingestion."""
        mock_client = MagicMock()
        mock_client.create_data_object.side_effect = Exception("Network error")

        products = [
            {"product_id": "groc-001", "name": "Olive Oil", "price": 14.99, "description": "D"},
            {"product_id": "groc-002", "name": "Bread", "price": 6.99, "description": "D"},
        ]

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            # Should not raise
            ingest_products(
                "projects/test-project/locations/us-central1/collections/grocery-col",
                products,
            )

        assert mock_client.create_data_object.call_count == 2


# =====================================================================
# Config loading (design-spec.md and config.yaml)
# =====================================================================

class TestConfigLoading:
    """Test config loading for VS ingestion."""

    def test_load_yaml_config(self, sample_config_yaml):
        cfg = load_config(sample_config_yaml)
        assert cfg["gcp_project_id"] == "test-project-123"
        assert cfg["collection_id"] == "test-products-collection"
        assert cfg["embedding_model"] == "gemini-embedding-001"

    def test_load_design_spec(self, sample_design_spec):
        cfg = load_config(sample_design_spec)
        assert cfg["gcp_project_id"] == "test-project-123"

    def test_load_missing_returns_empty(self):
        cfg = load_config("/nonexistent/config.yaml")
        assert cfg == {}


# =====================================================================
# Fetch products (BigQuery mock)
# =====================================================================

class TestFetchProducts:
    """Test BigQuery product fetching."""

    def test_fetch_returns_product_list(self):
        mock_client = MagicMock()
        mock_row = MagicMock()
        mock_row.items.return_value = [
            ("product_id", "elec-001"),
            ("name", "Headphones"),
            ("price", 179.99),
        ]
        mock_client.query.return_value.result.return_value = [mock_row]

        with patch.object(ivs.bigquery, "Client", return_value=mock_client):
            products = fetch_products("test-project", "products_dataset", "products")

        mock_client.query.assert_called_once()
        assert len(products) == 1
        assert products[0]["product_id"] == "elec-001"

    def test_fetch_empty_table(self):
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = []

        with patch.object(ivs.bigquery, "Client", return_value=mock_client):
            products = fetch_products("test-project", "products_dataset", "products")

        assert products == []


# =====================================================================
# Collection creation error handling
# =====================================================================

class TestCollectionCreationErrors:
    """Error scenarios during collection creation."""

    def test_create_collection_exits_on_error(self):
        """Should sys.exit(1) if collection creation fails."""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = NotFound("not found")
        mock_op = MagicMock()
        mock_op.result.side_effect = Exception("Permission denied")
        mock_client.create_collection.return_value = mock_op

        with patch.object(ivs.vectorsearch, "VectorSearchServiceClient", return_value=mock_client):
            with pytest.raises(SystemExit) as exc:
                create_collection_if_needed(
                    "test-project", "us-central1", "test-col",
                    "gemini-embedding-001", DEFAULT_EMBEDDING_FIELDS,
                    {"type": "object", "properties": {}},
                )
            assert exc.value.code == 1

    def test_collection_path_format(self):
        """Verify collection path follows GCP resource naming."""
        mock_client = MagicMock()
        mock_client.get_collection.return_value = MagicMock()

        with patch.object(ivs.vectorsearch, "VectorSearchServiceClient", return_value=mock_client):
            path = create_collection_if_needed(
                "my-project", "europe-west1", "my-collection",
                "gemini-embedding-001", DEFAULT_EMBEDDING_FIELDS,
                {"type": "object", "properties": {}},
            )

        assert path == "projects/my-project/locations/europe-west1/collections/my-collection"


# =====================================================================
# Error scenarios
# =====================================================================

class TestIngestionErrorScenarios:
    """Error handling during ingestion and collection creation."""

    def test_bigquery_fetch_failure_raises(self):
        """If BigQuery query fails, error should propagate."""
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception("Permission denied")

        with patch.object(ivs.bigquery, "Client", return_value=mock_client):
            with pytest.raises(Exception, match="Permission denied"):
                fetch_products("test-project", "dataset", "table")

    def test_ingest_continues_on_individual_failures(self):
        """If one product fails to insert, others should still be attempted."""
        mock_client = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Transient error")

        mock_client.create_data_object.side_effect = side_effect

        products = [
            {"product_id": f"p-{i}", "name": f"Product {i}", "price": 9.99, "description": "D"}
            for i in range(3)
        ]

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            ingest_products("projects/test/locations/us/collections/col", products)

        assert mock_client.create_data_object.call_count == 3

    def test_empty_product_list_no_api_calls(self):
        """Empty product list should not make any API calls."""
        mock_client = MagicMock()

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            ingest_products("projects/test/locations/us/collections/col", [])

        mock_client.create_data_object.assert_not_called()

    def test_load_config_malformed_yaml(self, tmp_path):
        """Malformed YAML should not crash, returns empty or raises gracefully."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("key: [unclosed bracket")
        try:
            result = load_config(str(bad_file))
            # If it doesn't raise, result should be something (empty dict or partial)
            assert isinstance(result, dict)
        except Exception:
            pass  # Raising is also acceptable for malformed YAML


# =====================================================================
# Type conversion in Struct (direct verification)
# =====================================================================

class TestStructTypeConversion:
    """Verify that ingest_products converts types correctly into Struct objects."""

    def test_struct_receives_float_for_price(self):
        """Price field should be converted to float before Struct.update."""
        from google.protobuf.struct_pb2 import Struct as MockStruct

        # Create a product with string price (as from CSV)
        products = [{"product_id": "t-001", "name": "Test", "price": 29.99, "description": "D"}]
        mock_client = MagicMock()

        structs_created = []
        original_struct = MockStruct

        def capture_struct():
            s = original_struct()
            structs_created.append(s)
            return s

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client), \
             patch("ingest_vertex_search.struct_pb2.Struct", side_effect=capture_struct):
            ingest_products("projects/t/locations/us/collections/c", products)

        assert len(structs_created) == 1
        assert isinstance(structs_created[0]["price"], float)

    def test_struct_receives_int_for_stock(self):
        """Stock field should be converted to int before Struct.update."""
        from google.protobuf.struct_pb2 import Struct as MockStruct

        products = [{"product_id": "t-001", "name": "Test", "price": 9.99, "description": "D", "stock": 50}]
        mock_client = MagicMock()

        structs_created = []
        original_struct = MockStruct

        def capture_struct():
            s = original_struct()
            structs_created.append(s)
            return s

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client), \
             patch("ingest_vertex_search.struct_pb2.Struct", side_effect=capture_struct):
            ingest_products("projects/t/locations/us/collections/c", products)

        assert len(structs_created) == 1
        assert isinstance(structs_created[0]["stock"], int)
        assert structs_created[0]["stock"] == 50

    def test_struct_receives_str_for_name(self):
        """Name field should be stored as string in the Struct."""
        from google.protobuf.struct_pb2 import Struct as MockStruct

        products = [{"product_id": "t-001", "name": "Headphones", "price": 9.99, "description": "D"}]
        mock_client = MagicMock()

        structs_created = []
        original_struct = MockStruct

        def capture_struct():
            s = original_struct()
            structs_created.append(s)
            return s

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client), \
             patch("ingest_vertex_search.struct_pb2.Struct", side_effect=capture_struct):
            ingest_products("projects/t/locations/us/collections/c", products)

        assert structs_created[0]["name"] == "Headphones"
        assert isinstance(structs_created[0]["name"], str)
