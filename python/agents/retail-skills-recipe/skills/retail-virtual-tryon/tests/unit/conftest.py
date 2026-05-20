"""Shared fixtures for retail virtual try-on unit tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock the SDKs before scripts import them at module load.
# google-genai is the new unified SDK (replaces vertexai.generative_models).
for mod in [
    "google", "google.cloud", "google.cloud.storage",
    "google.genai", "google.genai.types",
]:
    sys.modules.setdefault(mod, MagicMock())

# Make scripts/ importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def sample_design_spec(tmp_path):
    """A minimal design-spec.md with YAML frontmatter."""
    content = """---
gcp_project_id: "test-project"
gcp_region: "us-central1"
gemini_image_model: "flash"
safety_level: "block_most"
---

# Design Spec
"""
    spec = tmp_path / "design-spec.md"
    spec.write_text(content)
    return str(spec)
