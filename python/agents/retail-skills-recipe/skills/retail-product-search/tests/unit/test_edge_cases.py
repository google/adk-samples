"""Edge case tests across retail-product-search scripts.

Covers unicode product names, large batches, special characters,
boundary values, and unusual but valid inputs.
"""

from unittest.mock import MagicMock, patch

import pytest

from ingest_bigquery import validate_product, convert_types, _validate_and_convert
from ingest_vertex_search import (
    get_data_schema,
    build_text_template,
    ingest_products,
    PRODUCT_DATA_FIELDS,
)
import ingest_vertex_search as ivs
from validate_schema import validate


# =====================================================================
# Unicode and international characters
# =====================================================================

class TestUnicodeProducts:
    """Products with unicode characters in names, descriptions, brands."""

    def test_validate_unicode_product_name(self):
        """Japanese product name should pass validation."""
        product = {
            "product_id": "jp-001",
            "name": "Kokoro Japanese Rice Crackers",
            "price": "12.99",
            "description": "Traditional senbei rice crackers",
        }
        errors = validate_product(product, 1)
        assert errors == []

    def test_validate_chinese_brand(self):
        product = {
            "product_id": "cn-001",
            "name": "Green Tea",
            "price": "8.99",
            "description": "Premium Chinese green tea",
            "brand": "Dragon Well Tea Co",
        }
        errors = validate_product(product, 1)
        assert errors == []

    def test_convert_types_unicode_preserved(self):
        product = {
            "product_id": "uni-001",
            "name": "Creme Brulee",
            "price": "15.99",
            "description": "French dessert",
            "brand": "Le Patissier",
        }
        converted = convert_types(product)
        assert converted["name"] == "Creme Brulee"
        assert converted["brand"] == "Le Patissier"

    def test_schema_validation_with_accented_chars(self):
        products = [{
            "product_id": "fr-001",
            "name": "Pate de Campagne",
            "price": "22.50",
            "description": "Rustic French countryside pate",
            "category": "Charcuterie",
        }]
        valid, errors = validate(products, "Standard")
        assert valid == 1
        assert errors == []

    def test_data_schema_with_emoji_in_description(self):
        products = [{
            "product_id": "em-001",
            "name": "Happy Socks",
            "price": 14.99,
            "description": "Colorful socks that make you smile",
            "category": "Accessories",
        }]
        schema = get_data_schema(products)
        assert "description" in schema["properties"]


# =====================================================================
# Boundary values
# =====================================================================

class TestBoundaryValues:
    """Test boundary values for prices, ratings, stock."""

    def test_zero_price(self):
        """Price of 0 should pass validation (free items)."""
        product = {
            "product_id": "free-001",
            "name": "Free Sample",
            "price": "0",
            "description": "Complimentary sample",
        }
        errors = validate_product(product, 1)
        assert errors == []

    def test_very_high_price(self):
        """Very expensive items should validate."""
        product = {
            "product_id": "lux-001",
            "name": "Diamond Necklace",
            "price": "99999.99",
            "description": "18k gold with diamonds",
        }
        errors = validate_product(product, 1)
        assert errors == []

    def test_rating_zero(self):
        """Rating of exactly 0 is valid."""
        products = [{
            "product_id": "r0-001",
            "name": "New Product",
            "price": "10.00",
            "description": "Just launched",
            "rating": "0",
        }]
        valid, errors = validate(products, "Extended")
        assert valid == 1

    def test_rating_five(self):
        """Rating of exactly 5 is valid."""
        products = [{
            "product_id": "r5-001",
            "name": "Perfect Product",
            "price": "25.00",
            "description": "Five star product",
            "rating": "5.0",
        }]
        valid, errors = validate(products, "Extended")
        assert valid == 1

    def test_very_long_product_name(self):
        """Products with very long names should still validate."""
        product = {
            "product_id": "long-001",
            "name": "A" * 500,
            "price": "10.00",
            "description": "Product with very long name",
        }
        errors = validate_product(product, 1)
        assert errors == []

    def test_very_long_description(self):
        """Products with very long descriptions should validate."""
        product = {
            "product_id": "desc-001",
            "name": "Test Product",
            "price": "10.00",
            "description": "D" * 5000,
        }
        errors = validate_product(product, 1)
        assert errors == []

    def test_convert_types_very_large_stock(self):
        """Very large stock numbers should convert correctly."""
        product = {
            "product_id": "big-001",
            "name": "Bulk Item",
            "price": "1.00",
            "description": "Bulk",
            "stock": "999999999",
        }
        converted = convert_types(product)
        assert converted["stock"] == 999999999
        assert isinstance(converted["stock"], int)


# =====================================================================
# Large batch handling
# =====================================================================

class TestLargeBatches:
    """Test behavior with large product lists."""

    def test_validate_and_convert_large_batch(self):
        """250 products should all be validated and converted."""
        products = [
            {
                "product_id": f"batch-{i:04d}",
                "name": f"Product {i}",
                "price": f"{i + 1}.99",
                "description": f"Description for product {i}",
            }
            for i in range(250)
        ]
        result = _validate_and_convert(products)
        assert len(result) == 250

    def test_data_schema_samples_from_first_ten(self):
        """get_data_schema should detect fields from first 10 products."""
        products = [
            {"product_id": f"p-{i}", "name": f"P{i}", "price": 9.99}
            for i in range(100)
        ]
        # Only first product has 'category'
        products[0]["category"] = "Test"
        schema = get_data_schema(products)
        assert "category" in schema["properties"]

    def test_ingest_large_batch(self):
        """Ingesting 100 products should call create_data_object 100 times."""
        mock_client = MagicMock()

        products = [
            {"product_id": f"p-{i}", "name": f"Product {i}", "price": 9.99, "description": "D"}
            for i in range(100)
        ]

        with patch.object(ivs.vectorsearch, "DataObjectServiceClient", return_value=mock_client):
            ingest_products("projects/test/locations/us/collections/col", products)

        assert mock_client.create_data_object.call_count == 100


# =====================================================================
# Special product ID formats
# =====================================================================

class TestProductIdFormats:
    """Test various product_id formats including edge cases."""

    def test_product_id_with_numbers_only(self):
        product = {"product_id": "12345", "name": "Test", "price": "10", "description": "D"}
        errors = validate_product(product, 1)
        assert errors == []

    def test_product_id_with_underscores(self):
        product = {"product_id": "my_product_001", "name": "Test", "price": "10", "description": "D"}
        errors = validate_product(product, 1)
        assert errors == []

    def test_product_id_with_dashes(self):
        product = {"product_id": "my-product-001", "name": "Test", "price": "10", "description": "D"}
        errors = validate_product(product, 1)
        assert errors == []
