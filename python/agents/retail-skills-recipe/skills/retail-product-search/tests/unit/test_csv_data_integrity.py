"""Tests that verify real CSV test data files load correctly and have expected structure.

These tests use actual CSV files instead of fixtures, ensuring the test data
files themselves are valid and match the expected schema.
"""

import csv
from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"


class TestElectronicsCSV:
    """Verify electronics_products.csv integrity."""

    def test_file_exists(self):
        assert (TEST_DATA_DIR / "electronics_products.csv").exists()

    def test_row_count(self):
        with open(TEST_DATA_DIR / "electronics_products.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10

    def test_required_columns_present(self):
        with open(TEST_DATA_DIR / "electronics_products.csv") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        for col in ["product_id", "name", "price", "description"]:
            assert col in headers, f"Missing required column: {col}"

    def test_product_ids_unique(self):
        with open(TEST_DATA_DIR / "electronics_products.csv") as f:
            rows = list(csv.DictReader(f))
        ids = [r["product_id"] for r in rows]
        assert len(ids) == len(set(ids)), "Duplicate product_id found"

    def test_prices_are_numeric(self):
        with open(TEST_DATA_DIR / "electronics_products.csv") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            float(row["price"])  # Should not raise

    def test_all_product_ids_have_prefix(self):
        with open(TEST_DATA_DIR / "electronics_products.csv") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row["product_id"].startswith("elec-")


class TestFashionCSV:
    """Verify fashion_products.csv integrity."""

    def test_file_exists(self):
        assert (TEST_DATA_DIR / "fashion_products.csv").exists()

    def test_row_count(self):
        with open(TEST_DATA_DIR / "fashion_products.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10

    def test_special_characters_preserved(self):
        with open(TEST_DATA_DIR / "fashion_products.csv") as f:
            rows = list(csv.DictReader(f))
        brands = [r.get("brand", "") for r in rows]
        categories = [r.get("category", "") for r in rows]
        # Should have at least one special char (apostrophe or ampersand)
        all_text = " ".join(brands + categories)
        assert "'" in all_text or "&" in all_text


class TestGroceryCSV:
    """Verify grocery_products.csv integrity."""

    def test_file_exists(self):
        assert (TEST_DATA_DIR / "grocery_products.csv").exists()

    def test_row_count(self):
        with open(TEST_DATA_DIR / "grocery_products.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10

    def test_has_out_of_stock_items(self):
        with open(TEST_DATA_DIR / "grocery_products.csv") as f:
            rows = list(csv.DictReader(f))
        out_of_stock = [r for r in rows if r.get("stock") == "0"]
        assert len(out_of_stock) >= 1, "Expected at least one out-of-stock item"


class TestInvalidCSV:
    """Verify invalid_products.csv has expected invalid data."""

    def test_file_exists(self):
        assert (TEST_DATA_DIR / "invalid_products.csv").exists()

    def test_has_rows(self):
        with open(TEST_DATA_DIR / "invalid_products.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 7, "Expected at least 7 rows (6 invalid + 1 valid)"

    def test_has_valid_row(self):
        """At least one row should be fully valid (inv-007)."""
        with open(TEST_DATA_DIR / "invalid_products.csv") as f:
            rows = list(csv.DictReader(f))
        valid = [r for r in rows if r.get("product_id") == "inv-007"]
        assert len(valid) == 1
