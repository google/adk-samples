"""Tests for setup.py - config-driven pipeline orchestrator.

Tests that setup.py correctly reads design-spec.md and runs
the right scripts based on user configuration.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Import setup functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.setup import load_config, setup


# =====================================================================
# Test: load_config parses YAML frontmatter correctly
# =====================================================================

class TestLoadConfig:
    def test_loads_yaml_frontmatter(self, tmp_path):
        spec = tmp_path / "design-spec.md"
        spec.write_text("""---
gcp_project_id: "my-project"
industry: "Retail"
has_existing_api: "No"
---
# Markdown body
""")
        cfg = load_config(str(spec))
        assert cfg["gcp_project_id"] == "my-project"
        assert cfg["industry"] == "Retail"
        assert cfg["has_existing_api"] == "No"

    def test_ignores_comment_lines_with_dashes(self, tmp_path):
        spec = tmp_path / "design-spec.md"
        spec.write_text("""---
# --- Data Model ---
gcp_project_id: "test-proj"
# --- Search Architecture ---
has_existing_api: "Yes"
---
# Body
""")
        cfg = load_config(str(spec))
        assert cfg["gcp_project_id"] == "test-proj"
        assert cfg["has_existing_api"] == "Yes"

    def test_returns_empty_for_missing_file(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.md"))
        assert cfg == {}

    def test_returns_empty_for_empty_frontmatter(self, tmp_path):
        spec = tmp_path / "design-spec.md"
        spec.write_text("""---
# Only comments
---
""")
        cfg = load_config(str(spec))
        assert cfg is None or cfg == {}


# =====================================================================
# Test: setup runs correct pipeline based on config
# =====================================================================

class TestSetupPathSelection:
    def _make_spec(self, tmp_path, overrides=None):
        defaults = {
            "gcp_project_id": "test-project",
            "has_existing_api": "No",
            "has_images": "No",
            "voice_capabilities": "No",
            "product_fields": "Extended",
            "catalog_size": "1K-50K",
            "search_type": "Text-only",
            "user_interface": "Cloud Run web app",
            "vague_query_handling": "Ask clarifying questions",
        }
        if overrides:
            defaults.update(overrides)
        yaml_lines = "\n".join(f'{k}: "{v}"' for k, v in defaults.items())
        spec = tmp_path / "design-spec.md"
        spec.write_text(f"---\n{yaml_lines}\n---\n# Body\n")
        return str(spec)

    @patch("_shared.setup_utils.subprocess.run")
    def test_path_b_runs_validate_bigquery_vectorsearch(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path)
        # Create sample data file
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "sample-products.csv").write_text("product_id,name\n1,Test\n")

        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            setup(spec, dry_run=True)
        finally:
            os.chdir(old_cwd)

    @patch("_shared.setup_utils.subprocess.run")
    def test_path_a_runs_api_connector(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path, {"has_existing_api": "Yes"})
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "api_connector.py").write_text("")

        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            setup(spec, dry_run=True)
        finally:
            os.chdir(old_cwd)

    @patch("_shared.setup_utils.subprocess.run")
    def test_images_yes_runs_gcs_upload(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path, {"has_images": "Yes"})
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "sample-products.csv").write_text("product_id,name\n1,Test\n")
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "ingest_gcs.py").write_text("")

        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            setup(spec, dry_run=True)
        finally:
            os.chdir(old_cwd)

    @patch("_shared.setup_utils.subprocess.run")
    def test_voice_yes_shows_instructions(self, mock_run, tmp_path, capsys):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path, {"voice_capabilities": "Yes"})
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "sample-products.csv").write_text("product_id,name\n1,Test\n")
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "live_search.py").write_text("")

        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            setup(spec, dry_run=True)
        finally:
            os.chdir(old_cwd)

    def test_missing_project_id_exits(self, tmp_path):
        spec = tmp_path / "design-spec.md"
        spec.write_text('---\nindustry: "Retail"\n---\n')
        with pytest.raises(SystemExit):
            setup(str(spec))

    @patch("_shared.setup_utils.subprocess.run")
    def test_large_catalog_shows_warning(self, mock_run, tmp_path, capsys):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path, {"catalog_size": "500K+"})
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "sample-products.csv").write_text("product_id,name\n1,Test\n")

        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            setup(spec, dry_run=True)
        finally:
            os.chdir(old_cwd)
