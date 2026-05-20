"""Shared fixtures for retail content generation unit tests.

Provides mock GCP clients, mock Gemini client, sample product data
for 3 use cases (electronics, fashion, grocery), and reusable helpers.
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

for mod in [
    "google",
    "google.cloud",
    "google.cloud.bigquery",
    "google.cloud.storage",
    "google.api_core",
    "google.api_core.exceptions",
    "google.genai",
    "google.genai.types",
    "google.auth",
    "google.auth.transport",
    "google.auth.transport.requests",
]:
    _ensure_mock_module(mod)

# Make genai types work for GenerateContentConfig calls
_genai = sys.modules["google.genai"]
_genai.Client = MagicMock


class _MockGenerateContentConfig:
    """Records kwargs so tests can inspect temperature, max_output_tokens, etc."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# Set GenerateContentConfig on both the sys.modules entry AND the genai mock's
# .types attribute, so that `from google.genai import types` picks it up
_genai_types = sys.modules["google.genai.types"]
_genai_types.GenerateContentConfig = _MockGenerateContentConfig
_genai.types = _genai_types

# Add scripts directory to path so we can import the modules under test
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"

# ---------------------------------------------------------------------------
# Sample product data
# ---------------------------------------------------------------------------

@pytest.fixture
def electronics_products():
    """3 electronics products loaded from CSV."""
    with open(TEST_DATA_DIR / "electronics_products.csv") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def typed_electronics_products():
    """Electronics products with proper Python types (as from BigQuery)."""
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
def fashion_product():
    """A single fashion product for marketing copy tests."""
    return {
        "product_id": "fash-001",
        "name": "Cashmere Wrap Coat",
        "price": 895.00,
        "description": "Luxurious double-faced cashmere wrap coat with belt tie closure",
        "category": "Outerwear",
        "brand": "Brooks & Co",
        "image_url": "https://example.com/images/fash-001.jpg",
        "rating": 4.9,
        "stock": 8,
    }


@pytest.fixture
def grocery_product():
    """A single grocery product for translation tests."""
    return {
        "product_id": "groc-001",
        "name": "Organic Extra Virgin Olive Oil",
        "price": 14.99,
        "description": "Cold-pressed single-origin organic olive oil from Kalamata Greece 500ml",
        "category": "Oils & Vinegars",
        "brand": "OliveGrove",
        "rating": 4.8,
        "stock": 60,
    }


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def electronics_config():
    """Config for electronics store content generation."""
    return {
        "brand_tone": "Technical and enthusiastic",
        "include_specs": True,
        "include_use_cases": True,
        "description_length": "Medium (100-150 words)",
        "include_price_in_seo": False,
    }


@pytest.fixture
def fashion_config():
    """Config for fashion store with luxury brand voice."""
    return {
        "brand_tone": "Luxurious and aspirational",
        "brand_name": "Brooks & Co",
        "always_include": "craftsmanship, heritage",
        "never_use": "cheap, discount, bargain",
        "description_length": "Long (200-300 words)",
    }


@pytest.fixture
def grocery_config():
    """Config for grocery store translation and A/B tests."""
    return {
        "brand_tone": "Warm and wholesome",
        "always_include": "organic, fresh, natural",
        "description_length": "Short (50-75 words)",
    }


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a temporary config.yaml for testing."""
    import yaml
    config = {
        "gcp_project_id": "test-project-123",
        "brand_tone": "Professional and informative",
        "description_length": "Medium (100-150 words)",
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    return str(config_file)


@pytest.fixture
def sample_design_spec(tmp_path):
    """Create a temporary design-spec.md with YAML frontmatter."""
    content = """---
gcp_project_id: "test-project-123"
brand_tone: "Luxurious and aspirational"
brand_name: "Brooks & Co"
always_include: "craftsmanship, heritage"
never_use: "cheap, discount, bargain"
---

# Content Generation Design Spec
This is a test design spec for content generation.
"""
    spec_file = tmp_path / "design-spec.md"
    spec_file.write_text(content)
    return str(spec_file)


# ---------------------------------------------------------------------------
# Mock clients
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client with configurable generate_content response."""
    client = MagicMock()
    response = MagicMock()
    response.text = "Generated content goes here."
    client.models.generate_content.return_value = response
    return client


@pytest.fixture
def mock_bigquery_client():
    """Mock BigQuery client."""
    with patch("google.cloud.bigquery.Client") as MockClient:
        client = MockClient.return_value
        client.project = "test-project"
        yield client
