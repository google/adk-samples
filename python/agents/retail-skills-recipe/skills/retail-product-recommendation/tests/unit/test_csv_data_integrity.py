"""Tests that verify real CSV test data files load correctly and have expected structure.

These tests use actual CSV files instead of fixtures, ensuring the test data
files themselves are valid and match the expected schema.
"""

import csv
from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"


class TestEcommerceEventsCSV:
    """Verify ecommerce_events.csv integrity."""

    def test_file_exists(self):
        assert (TEST_DATA_DIR / "ecommerce_events.csv").exists()

    def test_row_count(self):
        with open(TEST_DATA_DIR / "ecommerce_events.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6

    def test_required_columns_present(self):
        with open(TEST_DATA_DIR / "ecommerce_events.csv") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        for col in ["event_id", "user_id", "product_id", "event_type", "timestamp"]:
            assert col in headers

    def test_event_ids_unique(self):
        with open(TEST_DATA_DIR / "ecommerce_events.csv") as f:
            rows = list(csv.DictReader(f))
        ids = [r["event_id"] for r in rows]
        assert len(ids) == len(set(ids))


class TestFashionEventsCSV:
    """Verify fashion_events.csv integrity."""

    def test_file_exists(self):
        assert (TEST_DATA_DIR / "fashion_events.csv").exists()

    def test_row_count(self):
        with open(TEST_DATA_DIR / "fashion_events.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6


class TestGroceryEventsCSV:
    """Verify grocery_events.csv integrity."""

    def test_file_exists(self):
        assert (TEST_DATA_DIR / "grocery_events.csv").exists()

    def test_row_count(self):
        with open(TEST_DATA_DIR / "grocery_events.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6


class TestInvalidEventsCSV:
    """Verify invalid_events.csv has expected structure."""

    def test_file_exists(self):
        assert (TEST_DATA_DIR / "invalid_events.csv").exists()

    def test_has_valid_row(self):
        with open(TEST_DATA_DIR / "invalid_events.csv") as f:
            rows = list(csv.DictReader(f))
        valid = [r for r in rows if r.get("event_id") == "evt-good"]
        assert len(valid) == 1
