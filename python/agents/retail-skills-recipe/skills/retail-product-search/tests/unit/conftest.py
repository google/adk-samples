"""Shared fixtures for retail product search unit tests.

Provides mock GCP clients, sample product data for 3 use cases
(electronics, fashion, grocery), and reusable test helpers.
"""

import csv
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock GCP modules before any script imports them at module level
# ---------------------------------------------------------------------------

def _ensure_mock_module(name):
    """Create a mock module at `name` if it doesn't exist yet."""
    if name not in sys.modules:
        sys.modules[name] = MagicMock()

# google.cloud.* hierarchy
for mod in [
    "google",
    "google.cloud",
    "google.cloud.bigquery",
    "google.cloud.storage",
    "google.cloud.vectorsearch",
    "google.cloud.aiplatform",
    "google.cloud.pubsub_v1",
    "google.protobuf",
    "google.protobuf.struct_pb2",
    "google.auth",
    "google.auth.transport",
    "google.auth.transport.requests",
    "vertexai",
    "vertexai.language_models",
    "vertexai.vision_models",
    "functions_framework",
]:
    _ensure_mock_module(mod)

# Create real exception classes so `except` clauses work
class NotFound(Exception):
    pass

class AlreadyExists(Exception):
    pass

# Create a proper exceptions module (not a MagicMock) so `except` works
_exceptions_mod = types.ModuleType("google.api_core.exceptions")
_exceptions_mod.NotFound = NotFound
_exceptions_mod.AlreadyExists = AlreadyExists
sys.modules["google.api_core"] = types.ModuleType("google.api_core")
sys.modules["google.api_core"].exceptions = _exceptions_mod
sys.modules["google.api_core.exceptions"] = _exceptions_mod

# Make vectorsearch have the classes we need
_vs = sys.modules["google.cloud.vectorsearch"]
_vs.VectorSearchServiceClient = MagicMock
_vs.DataObjectServiceClient = MagicMock
_vs.DataObjectSearchServiceClient = MagicMock
_vs.GetCollectionRequest = MagicMock
_vs.CreateCollectionRequest = MagicMock
_vs.DeleteCollectionRequest = MagicMock
_vs.BatchCreateDataObjectsRequest = MagicMock
_vs.DataObject = MagicMock
_vs.UpsertDataObjectRequest = MagicMock
_vs.DeleteDataObjectRequest = MagicMock
_vs.SearchDataObjectsRequest = MagicMock
_vs.SemanticSearch = MagicMock
_vs.OutputFields = MagicMock

# Make struct_pb2.Struct behave like a dict but enforce type validation
# so tests fail if code passes raw dicts instead of Struct objects
class MockStruct(dict):
    """Mock that mimics google.protobuf.struct_pb2.Struct with type enforcement.

    Real Struct objects only accept certain types (str, int, float, bool, None,
    list, dict). This mock validates types on update to catch code that passes
    raw dicts where a Struct is expected.
    """
    _ALLOWED_VALUE_TYPES = (str, int, float, bool, type(None), list, dict)

    def update(self, *args, **kwargs):
        if args:
            items = args[0].items() if isinstance(args[0], dict) else args[0]
            for key, value in items:
                if not isinstance(key, str):
                    raise TypeError(
                        f"Struct keys must be strings, got {type(key).__name__}"
                    )
                if not isinstance(value, self._ALLOWED_VALUE_TYPES):
                    raise TypeError(
                        f"Struct value for '{key}' must be one of "
                        f"{[t.__name__ for t in self._ALLOWED_VALUE_TYPES]}, "
                        f"got {type(value).__name__}"
                    )
        super().update(*args, **kwargs)

sys.modules["google.protobuf.struct_pb2"].Struct = MockStruct

# Add scripts directory to path so we can import the modules under test
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"

# ---------------------------------------------------------------------------
# Sample product data for each use case
# ---------------------------------------------------------------------------

@pytest.fixture
def electronics_products():
    """10 electronics store products (headphones, keyboards, cables, etc.)."""
    with open(TEST_DATA_DIR / "electronics_products.csv") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def fashion_products():
    """10 fashion store products (shirts, jeans, boots, bags, etc.)."""
    with open(TEST_DATA_DIR / "fashion_products.csv") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def grocery_products():
    """10 grocery/organic food products (olive oil, bread, eggs, etc.).
    Includes 2 out-of-stock items (groc-002, groc-007).
    """
    with open(TEST_DATA_DIR / "grocery_products.csv") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def invalid_products():
    """Products with various validation errors (missing fields, bad types)."""
    with open(TEST_DATA_DIR / "invalid_products.csv") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def electronics_products_json():
    """3 electronics products loaded from JSON format."""
    with open(TEST_DATA_DIR / "electronics_products.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Product dicts with proper types (as they'd come from BigQuery)
# ---------------------------------------------------------------------------

@pytest.fixture
def typed_electronics_products():
    """Electronics products with proper Python types (not CSV strings)."""
    return [
        {
            "product_id": "elec-001",
            "name": "Wireless Noise-Cancelling Headphones",
            "price": 179.99,
            "description": "Over-ear Bluetooth headphones with active noise cancellation and 35-hour battery life",
            "category": "Audio",
            "brand": "SoundMax",
            "image_url": "https://example.com/images/elec-001.jpg",
            "rating": 4.6,
            "stock": 45,
        },
        {
            "product_id": "elec-002",
            "name": "USB-C Docking Station",
            "price": 89.99,
            "description": "12-in-1 USB-C hub with dual HDMI 4K 60Hz and 100W passthrough charging",
            "category": "Accessories",
            "brand": "TechHub",
            "rating": 4.3,
            "stock": 120,
        },
        {
            "product_id": "elec-003",
            "name": "Mechanical Gaming Keyboard",
            "price": 149.99,
            "description": "Full-size RGB mechanical keyboard with Cherry MX Brown switches and wrist rest",
            "category": "Peripherals",
            "brand": "KeyForce",
            "image_url": "https://example.com/images/elec-003.jpg",
            "rating": 4.8,
            "stock": 25,
        },
    ]


@pytest.fixture
def typed_grocery_products():
    """Grocery products with proper types, including out-of-stock items."""
    return [
        {
            "product_id": "groc-001",
            "name": "Organic Extra Virgin Olive Oil",
            "price": 14.99,
            "description": "Cold-pressed single-origin organic olive oil from Kalamata Greece 500ml",
            "category": "Oils & Vinegars",
            "brand": "OliveGrove",
            "rating": 4.8,
            "stock": 60,
        },
        {
            "product_id": "groc-002",
            "name": "Sourdough Bread Loaf",
            "price": 6.99,
            "description": "Artisan sourdough bread baked fresh daily with organic flour",
            "category": "Bakery",
            "brand": "FreshOven",
            "rating": 4.5,
            "stock": 0,  # Out of stock
        },
        {
            "product_id": "groc-008",
            "name": "Wild-Caught Salmon Fillet",
            "price": 18.99,
            "description": "Fresh Atlantic wild-caught salmon fillet skin-on per pound",
            "category": "Seafood",
            "brand": "OceanFresh",
            "rating": 4.9,
            "stock": 10,
        },
    ]


# ---------------------------------------------------------------------------
# Mock GCP clients
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_bigquery_client():
    """Mock BigQuery client that returns configurable query results."""
    with patch("google.cloud.bigquery.Client") as MockClient:
        client = MockClient.return_value
        client.project = "test-project"
        yield client


@pytest.fixture
def mock_vectorsearch_service_client():
    """Mock VectorSearchServiceClient for collection creation."""
    with patch("google.cloud.vectorsearch.VectorSearchServiceClient") as MockClient:
        client = MockClient.return_value
        yield client


@pytest.fixture
def mock_dataobject_client():
    """Mock DataObjectServiceClient for product ingestion."""
    with patch("google.cloud.vectorsearch.DataObjectServiceClient") as MockClient:
        client = MockClient.return_value
        yield client


@pytest.fixture
def mock_dataobject_search_client():
    """Mock DataObjectSearchServiceClient for product search."""
    with patch("google.cloud.vectorsearch.DataObjectSearchServiceClient") as MockClient:
        client = MockClient.return_value
        yield client


@pytest.fixture
def mock_storage_client():
    """Mock Cloud Storage client."""
    with patch("google.cloud.storage.Client") as MockClient:
        client = MockClient.return_value
        yield client


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a temporary config.yaml for testing."""
    config = {
        "gcp_project_id": "test-project-123",
        "gcp_region": "us-central1",
        "project_name": "electronics-search",
        "embedding_model": "gemini-embedding-001",
        "embedding_fields": "name, description, category, brand",
        "collection_id": "test-products-collection",
    }
    config_file = tmp_path / "config.yaml"
    import yaml
    config_file.write_text(yaml.dump(config))
    return str(config_file)


@pytest.fixture
def sample_design_spec(tmp_path):
    """Create a temporary design-spec.md with YAML frontmatter."""
    content = """---
gcp_project_id: "test-project-123"
gcp_region: "us-central1"
project_name: "electronics-search"
embedding_model: "gemini-embedding-001"
embedding_fields: "name, description, category, brand"
collection_id: "test-products-collection"
---

# Design Spec
This is a test design spec.
"""
    spec_file = tmp_path / "design-spec.md"
    spec_file.write_text(content)
    return str(spec_file)
