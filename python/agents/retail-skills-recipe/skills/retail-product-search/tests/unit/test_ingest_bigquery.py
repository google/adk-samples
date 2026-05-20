"""Tests for ingest_bigquery.py across 3 use cases.

Use Case 1 (Electronics): Standard CSV ingestion, type conversion
Use Case 2 (Fashion): Products with special characters, optional fields
Use Case 3 (Grocery): Out-of-stock products, low prices, config loading
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from ingest_bigquery import (
    validate_product,
    convert_types,
    load_from_csv,
    load_from_json,
    _validate_and_convert,
    ingest,
    load_config,
    REQUIRED_FIELDS,
    OPTIONAL_FIELDS,
    SCHEMA,
)

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"


# =====================================================================
# Use Case 1: Electronics Store — CSV ingestion & type conversion
# =====================================================================

class TestElectronicsIngestion:
    """Electronics: standard CSV, all fields populated, proper types."""

    def test_validate_valid_product(self):
        product = {
            "product_id": "elec-001",
            "name": "Wireless Headphones",
            "price": "179.99",
            "description": "Over-ear headphones",
            "category": "Audio",
            "brand": "SoundMax",
        }
        errors = validate_product(product, 1)
        assert errors == []

    def test_convert_types_numeric_fields(self):
        product = {
            "product_id": "elec-001",
            "name": "Wireless Headphones",
            "price": "179.99",
            "description": "Over-ear headphones",
            "category": "Audio",
            "brand": "SoundMax",
            "rating": "4.6",
            "stock": "45",
        }
        converted = convert_types(product)
        assert converted["price"] == 179.99
        assert isinstance(converted["price"], float)
        assert converted["rating"] == 4.6
        assert isinstance(converted["rating"], float)
        assert converted["stock"] == 45
        assert isinstance(converted["stock"], int)

    def test_convert_types_preserves_strings(self):
        product = {
            "product_id": "elec-003",
            "name": "Mechanical Keyboard",
            "price": "149.99",
            "description": "Full-size RGB keyboard",
            "category": "Peripherals",
            "brand": "KeyForce",
            "image_url": "https://example.com/elec-003.jpg",
        }
        converted = convert_types(product)
        assert converted["product_id"] == "elec-003"
        assert converted["name"] == "Mechanical Keyboard"
        assert converted["category"] == "Peripherals"
        assert converted["image_url"] == "https://example.com/elec-003.jpg"

    def test_load_csv_local_file(self):
        """Load electronics CSV — should get 10 valid products."""
        with patch("ingest_bigquery.storage"):
            products = load_from_csv(str(TEST_DATA_DIR / "electronics_products.csv"))
        assert len(products) == 10
        assert products[0]["product_id"] == "elec-001"
        assert isinstance(products[0]["price"], float)

    def test_validate_and_convert_all_electronics(self, electronics_products):
        """All 10 electronics products should pass validation."""
        products = _validate_and_convert(electronics_products)
        assert len(products) == 10

    @patch("ingest_bigquery.bigquery")
    def test_ingest_calls_bigquery(self, mock_bq_module):
        """Verify ingest() creates dataset/table and inserts rows."""
        mock_client = MagicMock()
        mock_bq_module.Client.return_value = mock_client
        mock_client.get_dataset.side_effect = Exception("Not found")
        mock_client.get_table.side_effect = Exception("Not found")
        mock_client.insert_rows_json.return_value = []

        with patch("ingest_bigquery.load_from_csv") as mock_load:
            mock_load.return_value = [
                {"product_id": "elec-001", "name": "Headphones", "price": 179.99, "description": "Test"},
            ]
            ingest("test-project", "products_dataset", "products",
                   str(TEST_DATA_DIR / "electronics_products.csv"), "csv")

        mock_client.create_dataset.assert_called_once()
        mock_client.create_table.assert_called_once()
        mock_client.insert_rows_json.assert_called_once()


# =====================================================================
# Use Case 2: Fashion Store — special characters, optional fields
# =====================================================================

class TestFashionIngestion:
    """Fashion: apostrophes in categories, missing optional fields."""

    def test_validate_product_with_apostrophe_category(self):
        product = {
            "product_id": "fash-001",
            "name": "Classic Fit Oxford Shirt",
            "price": "59.99",
            "description": "100% cotton button-down",
            "category": "Men's Shirts",
            "brand": "Brooks & Co",
        }
        errors = validate_product(product, 1)
        assert errors == []

    def test_validate_product_missing_optional_image(self):
        """Fashion products without image_url should still validate."""
        product = {
            "product_id": "fash-005",
            "name": "Canvas Tote Bag",
            "price": "34.99",
            "description": "Heavy-duty canvas tote",
            "category": "Bags",
            "brand": "EcoCarry",
        }
        errors = validate_product(product, 1)
        assert errors == []

    def test_convert_preserves_ampersand_brand(self):
        product = {
            "product_id": "fash-001",
            "name": "Oxford Shirt",
            "price": "59.99",
            "description": "Cotton shirt",
            "brand": "Brooks & Co",
        }
        converted = convert_types(product)
        assert converted["brand"] == "Brooks & Co"

    def test_load_fashion_csv(self):
        with patch("ingest_bigquery.storage"):
            products = load_from_csv(str(TEST_DATA_DIR / "fashion_products.csv"))
        assert len(products) == 10
        # Verify brand with special char survived
        shirt = [p for p in products if p["product_id"] == "fash-001"][0]
        assert shirt["brand"] == "Brooks & Co"

    def test_convert_empty_optional_fields(self):
        """Products with empty optional fields should convert cleanly."""
        product = {
            "product_id": "fash-010",
            "name": "Crew Socks 6-Pack",
            "price": "24.99",
            "description": "Combed cotton socks",
            "image_url": "",
        }
        converted = convert_types(product)
        assert converted["image_url"] == ""


# =====================================================================
# Use Case 3: Grocery Store — out-of-stock, low prices, config
# =====================================================================

class TestGroceryIngestion:
    """Grocery: perishables, stock=0, decimal prices, config loading."""

    def test_validate_out_of_stock_product(self):
        """Products with stock=0 should still pass validation."""
        product = {
            "product_id": "groc-002",
            "name": "Sourdough Bread",
            "price": "6.99",
            "description": "Artisan sourdough",
            "category": "Bakery",
            "brand": "FreshOven",
            "stock": "0",
        }
        errors = validate_product(product, 1)
        assert errors == []

    def test_convert_zero_stock(self):
        """stock=0 is falsy but should not be treated as missing."""
        product = {
            "product_id": "groc-007",
            "name": "Baby Spinach",
            "price": "3.99",
            "description": "Organic spinach",
            "stock": "0",
        }
        converted = convert_types(product)
        # Note: stock="0" is truthy (non-empty string), so it converts
        assert "stock" in converted
        assert converted["stock"] == 0

    def test_validate_low_price(self):
        product = {
            "product_id": "groc-007",
            "name": "Baby Spinach",
            "price": "3.99",
            "description": "Organic baby spinach leaves",
        }
        errors = validate_product(product, 1)
        assert errors == []

    def test_load_grocery_csv(self):
        with patch("ingest_bigquery.storage"):
            products = load_from_csv(str(TEST_DATA_DIR / "grocery_products.csv"))
        assert len(products) == 10
        # Verify out-of-stock items loaded
        bread = [p for p in products if p["product_id"] == "groc-002"][0]
        assert bread["stock"] == 0

    def test_load_json_format(self):
        """Test JSON loading with properly typed numeric values."""
        with patch("ingest_bigquery.storage"):
            products = load_from_json(str(TEST_DATA_DIR / "electronics_products.json"))
        assert len(products) == 3
        assert isinstance(products[0]["price"], float)


# =====================================================================
# Config loading
# =====================================================================

class TestConfigLoading:
    """Test config.yaml and design-spec.md loading."""

    def test_load_yaml_config(self, sample_config_yaml):
        cfg = load_config(sample_config_yaml)
        assert cfg["gcp_project_id"] == "test-project-123"
        assert cfg["gcp_region"] == "us-central1"

    def test_load_design_spec_frontmatter(self, sample_design_spec):
        cfg = load_config(sample_design_spec)
        assert cfg["gcp_project_id"] == "test-project-123"

    def test_load_missing_config_returns_empty(self):
        cfg = load_config("/nonexistent/path/config.yaml")
        assert cfg == {}


# =====================================================================
# Validation edge cases
# =====================================================================

class TestValidationEdgeCases:
    """Edge cases in product validation."""

    def test_missing_required_field(self):
        product = {"name": "No ID", "price": "10", "description": "Missing product_id"}
        errors = validate_product(product, 1)
        assert any("product_id" in e for e in errors)

    def test_empty_required_field(self):
        product = {"product_id": "", "name": "Empty ID", "price": "10", "description": "Empty ID"}
        errors = validate_product(product, 1)
        assert any("product_id" in e for e in errors)

    def test_non_numeric_price(self):
        product = {"product_id": "p1", "name": "Bad", "price": "free", "description": "Free item"}
        errors = validate_product(product, 1)
        assert any("numeric" in e for e in errors)

    def test_non_integer_stock(self):
        product = {"product_id": "p1", "name": "Bad", "price": "10", "description": "D", "stock": "1.5"}
        errors = validate_product(product, 1)
        assert any("integer" in e for e in errors)

    def test_validate_and_convert_skips_invalid_rows(self, invalid_products):
        """Invalid rows should be skipped, valid ones converted."""
        products = _validate_and_convert(invalid_products)
        # Only row 7 (inv-007) is fully valid
        assert len(products) >= 1
        assert all(isinstance(p["price"], float) for p in products)


# =====================================================================
# Schema constants
# =====================================================================

class TestSchemaConstants:
    """Verify schema constant definitions."""

    def test_required_fields_count(self):
        assert len(REQUIRED_FIELDS) == 4

    def test_required_fields_content(self):
        assert "product_id" in REQUIRED_FIELDS
        assert "name" in REQUIRED_FIELDS
        assert "price" in REQUIRED_FIELDS
        assert "description" in REQUIRED_FIELDS

    def test_optional_fields_content(self):
        assert "category" in OPTIONAL_FIELDS
        assert "brand" in OPTIONAL_FIELDS
        assert "image_url" in OPTIONAL_FIELDS
        assert "rating" in OPTIONAL_FIELDS
        assert "stock" in OPTIONAL_FIELDS

    def test_schema_has_nine_fields(self):
        assert len(SCHEMA) == 9
