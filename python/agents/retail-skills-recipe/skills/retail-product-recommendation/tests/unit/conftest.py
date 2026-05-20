"""Shared fixtures for retail product recommendation unit tests.

Provides mock GCP clients, sample event data for 3 use cases
(e-commerce, fashion, grocery), and reusable test helpers.
"""

import csv
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
]:
    _ensure_mock_module(mod)

# Add scripts directory to path so we can import the modules under test
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"

# ---------------------------------------------------------------------------
# Event data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ecommerce_events_csv():
    """Path to e-commerce browse-to-purchase events CSV."""
    return str(TEST_DATA_DIR / "ecommerce_events.csv")


@pytest.fixture
def fashion_events_csv():
    """Path to fashion discovery events CSV."""
    return str(TEST_DATA_DIR / "fashion_events.csv")


@pytest.fixture
def grocery_events_csv():
    """Path to grocery repeat-customer events CSV."""
    return str(TEST_DATA_DIR / "grocery_events.csv")


@pytest.fixture
def invalid_events_csv():
    """Path to events with various validation errors."""
    return str(TEST_DATA_DIR / "invalid_events.csv")


@pytest.fixture
def ecommerce_events():
    """E-commerce events loaded as list of dicts (raw CSV rows)."""
    with open(TEST_DATA_DIR / "ecommerce_events.csv") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def fashion_events():
    """Fashion events loaded as list of dicts (raw CSV rows)."""
    with open(TEST_DATA_DIR / "fashion_events.csv") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def grocery_events():
    """Grocery events loaded as list of dicts (raw CSV rows)."""
    with open(TEST_DATA_DIR / "grocery_events.csv") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def invalid_events():
    """Invalid events loaded as list of dicts (raw CSV rows)."""
    with open(TEST_DATA_DIR / "invalid_events.csv") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a temporary config.yaml for testing."""
    import yaml
    config = {
        "gcp_project_id": "test-project-123",
        "gcp_region": "us-central1",
        "dataset_id": "products_dataset",
        "table_id": "user_events",
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    return str(config_file)


@pytest.fixture
def sample_design_spec(tmp_path):
    """Create a temporary design-spec.md with YAML frontmatter."""
    content = """---
gcp_project_id: "test-project-123"
gcp_region: "us-central1"
dataset_id: "products_dataset"
table_id: "user_events"
---

# Design Spec
This is a test design spec for product recommendations.
"""
    spec_file = tmp_path / "design-spec.md"
    spec_file.write_text(content)
    return str(spec_file)
