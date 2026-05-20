"""Tests for validate_schema.py across 3 use cases.

Use Case 1 (Electronics): Standard product catalog, all fields populated
Use Case 2 (Fashion): Products with missing optional fields (image_url)
Use Case 3 (Grocery): Products with out-of-stock (stock=0), edge-case ratings
"""

import json
import csv
from pathlib import Path

import pytest

from validate_schema import load_records, validate, FIELD_LEVELS


TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"


# =====================================================================
# Use Case 1: Electronics Store
# =====================================================================

class TestElectronicsValidation:
    """Electronics store: 10 products, well-formed data, all field levels."""

    def test_all_valid_at_extended_level(self, electronics_products):
        valid, errors = validate(electronics_products, "Extended")
        assert valid == 10
        assert errors == []

    def test_all_valid_at_basic_level(self, electronics_products):
        valid, errors = validate(electronics_products, "Basic")
        # Basic level doesn't recognize category/brand/etc as valid fields
        assert valid == 0  # All rows have "unexpected fields"
        assert any("unexpected fields" in e for e in errors)

    def test_all_valid_at_standard_level(self, electronics_products):
        valid, errors = validate(electronics_products, "Standard")
        # Standard doesn't include rating/stock, so they're unexpected
        assert all("unexpected fields" in e for e in errors)

    def test_load_csv_format(self):
        records = load_records(TEST_DATA_DIR / "electronics_products.csv")
        assert len(records) == 10
        assert records[0]["product_id"] == "elec-001"
        assert records[0]["name"] == "Wireless Noise-Cancelling Headphones"

    def test_load_json_format(self):
        records = load_records(TEST_DATA_DIR / "electronics_products.json")
        assert len(records) == 3
        assert records[0]["product_id"] == "elec-001"

    def test_price_is_numeric(self, electronics_products):
        """All electronics prices should pass numeric validation."""
        valid, errors = validate(electronics_products, "Extended")
        assert not any("price" in e and "numeric" in e for e in errors)

    def test_ratings_in_range(self, electronics_products):
        """All electronics ratings should be 0-5."""
        valid, errors = validate(electronics_products, "Extended")
        assert not any("rating" in e and "0-5" in e for e in errors)

    def test_stock_is_integer(self, electronics_products):
        """All electronics stock values should be valid integers."""
        valid, errors = validate(electronics_products, "Extended")
        assert not any("stock" in e and "integer" in e for e in errors)


# =====================================================================
# Use Case 2: Fashion Store
# =====================================================================

class TestFashionValidation:
    """Fashion store: products with apostrophes in categories, missing images."""

    def test_all_valid_at_extended_level(self, fashion_products):
        valid, errors = validate(fashion_products, "Extended")
        assert valid == 10
        assert errors == []

    def test_missing_image_url_is_ok(self, fashion_products):
        """Products without image_url should still be valid (it's optional)."""
        products_without_images = [
            p for p in fashion_products if not p.get("image_url")
        ]
        assert len(products_without_images) == 2  # fash-005, fash-010
        valid, errors = validate(products_without_images, "Extended")
        assert valid == 2
        assert errors == []

    def test_special_chars_in_category(self, fashion_products):
        """Categories like "Men's Shirts" with apostrophes should validate."""
        mens_shirt = [p for p in fashion_products if p["product_id"] == "fash-001"][0]
        assert mens_shirt["category"] == "Men's Shirts"
        valid, errors = validate([mens_shirt], "Extended")
        assert valid == 1

    def test_high_price_products(self, fashion_products):
        """Fashion items over $100 should validate fine."""
        expensive = [p for p in fashion_products if float(p["price"]) > 100]
        assert len(expensive) >= 3  # boots, cardigan, blazer, etc.
        valid, errors = validate(expensive, "Extended")
        assert valid == len(expensive)


# =====================================================================
# Use Case 3: Grocery Store
# =====================================================================

class TestGroceryValidation:
    """Grocery store: perishables, out-of-stock items, decimal prices."""

    def test_all_valid_at_extended_level(self, grocery_products):
        valid, errors = validate(grocery_products, "Extended")
        assert valid == 10
        assert errors == []

    def test_out_of_stock_validates(self, grocery_products):
        """Products with stock=0 should still pass validation."""
        out_of_stock = [p for p in grocery_products if p["stock"] == "0"]
        assert len(out_of_stock) == 2  # groc-002 (bread), groc-007 (spinach)
        valid, errors = validate(out_of_stock, "Extended")
        assert valid == 2
        assert errors == []

    def test_low_price_products(self, grocery_products):
        """Grocery items with prices like $3.99, $4.99 should validate."""
        cheap = [p for p in grocery_products if float(p["price"]) < 10]
        assert len(cheap) >= 3
        valid, errors = validate(cheap, "Extended")
        assert valid == len(cheap)

    def test_full_level_rejects_unknown_fields(self):
        """Full level should reject fields not in the Full schema."""
        products = [{
            "product_id": "groc-test",
            "name": "Test",
            "price": "1.99",
            "description": "Test product",
            "category": "Test",
            "organic_certified": "yes",  # Not a recognized field
        }]
        valid, errors = validate(products, "Full")
        assert valid == 0
        assert any("unexpected fields" in e for e in errors)


# =====================================================================
# Cross-cutting: Invalid data
# =====================================================================

class TestInvalidProducts:
    """Products with various validation errors."""

    def test_missing_product_id(self, invalid_products):
        """Row 1 has empty product_id."""
        valid, errors = validate(invalid_products, "Extended")
        assert any("Row 1" in e and "product_id" in e for e in errors)

    def test_missing_name(self, invalid_products):
        """Row 2 has empty name."""
        valid, errors = validate(invalid_products, "Extended")
        assert any("Row 2" in e and "name" in e for e in errors)

    def test_missing_price(self, invalid_products):
        """Row 3 has empty price."""
        valid, errors = validate(invalid_products, "Extended")
        assert any("Row 3" in e and "price" in e for e in errors)

    def test_non_numeric_price(self, invalid_products):
        """Row 4 has 'not_a_number' as price."""
        valid, errors = validate(invalid_products, "Extended")
        assert any("Row 4" in e and "numeric" in e for e in errors)

    def test_rating_out_of_range(self, invalid_products):
        """Row 5 has rating=6.5 (above 5)."""
        valid, errors = validate(invalid_products, "Extended")
        assert any("Row 5" in e and "0-5" in e for e in errors)

    def test_non_integer_stock(self, invalid_products):
        """Row 6 has stock='ten' (non-integer)."""
        valid, errors = validate(invalid_products, "Extended")
        assert any("Row 6" in e and "integer" in e for e in errors)

    def test_valid_row_passes(self, invalid_products):
        """Row 7 is valid and should be counted."""
        valid, errors = validate(invalid_products, "Extended")
        assert valid >= 1  # At least row 7 is valid

    def test_negative_rating(self, invalid_products):
        """Row 8 has rating=-1.0 (below 0)."""
        valid, errors = validate(invalid_products, "Extended")
        assert any("Row 8" in e and "0-5" in e for e in errors)

    def test_total_error_count(self, invalid_products):
        """Should have errors for rows 1-6 and 8, only row 7 valid."""
        valid, errors = validate(invalid_products, "Extended")
        assert valid == 1  # Only row 7
        assert len(errors) >= 7  # At least one error per invalid row


# =====================================================================
# Field level definitions
# =====================================================================

class TestFieldLevels:
    """Verify field level definitions are correct."""

    def test_basic_has_four_required(self):
        assert FIELD_LEVELS["Basic"]["required"] == [
            "product_id", "name", "price", "description"
        ]
        assert FIELD_LEVELS["Basic"]["optional"] == []

    def test_standard_adds_category_brand_image(self):
        assert "category" in FIELD_LEVELS["Standard"]["optional"]
        assert "brand" in FIELD_LEVELS["Standard"]["optional"]
        assert "image_url" in FIELD_LEVELS["Standard"]["optional"]

    def test_extended_adds_rating_stock(self):
        assert "rating" in FIELD_LEVELS["Extended"]["optional"]
        assert "stock" in FIELD_LEVELS["Extended"]["optional"]

    def test_full_adds_variants_tags(self):
        assert "variants" in FIELD_LEVELS["Full"]["optional"]
        assert "tags" in FIELD_LEVELS["Full"]["optional"]
        assert "specifications" in FIELD_LEVELS["Full"]["optional"]

    def test_all_levels_share_same_required(self):
        required = ["product_id", "name", "price", "description"]
        for level in FIELD_LEVELS:
            assert FIELD_LEVELS[level]["required"] == required


# =====================================================================
# File format loading
# =====================================================================

class TestLoadRecords:
    """Test loading from various file formats."""

    def test_load_csv(self):
        records = load_records(TEST_DATA_DIR / "electronics_products.csv")
        assert len(records) == 10

    def test_load_json_array(self):
        records = load_records(TEST_DATA_DIR / "electronics_products.json")
        assert len(records) == 3

    def test_load_json_with_products_key(self, tmp_path):
        data = {"products": [
            {"product_id": "p1", "name": "Test", "price": 9.99, "description": "Test"},
        ]}
        f = tmp_path / "products.json"
        f.write_text(json.dumps(data))
        records = load_records(f)
        assert len(records) == 1

    def test_load_jsonl(self, tmp_path):
        lines = [
            json.dumps({"product_id": "p1", "name": "Test", "price": 9.99, "description": "A"}),
            json.dumps({"product_id": "p2", "name": "Test2", "price": 19.99, "description": "B"}),
        ]
        f = tmp_path / "products.jsonl"
        f.write_text("\n".join(lines))
        records = load_records(f)
        assert len(records) == 2

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "products.xml"
        f.write_text("<products></products>")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_records(f)
