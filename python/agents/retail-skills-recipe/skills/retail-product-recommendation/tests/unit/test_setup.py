"""Tests for recommendation setup.py - config-driven pipeline orchestrator."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.setup import load_config, setup


class TestLoadConfig:
    def test_loads_frontmatter(self, tmp_path):
        spec = tmp_path / "design-spec.md"
        spec.write_text('---\ngcp_project_id: "proj"\nrecommendation_type: "collaborative"\n---\n')
        cfg = load_config(str(spec))
        assert cfg["gcp_project_id"] == "proj"
        assert cfg["recommendation_type"] == "collaborative"


class TestSetupRecommendation:
    def _make_spec(self, tmp_path, overrides=None):
        defaults = {
            "gcp_project_id": "test-project",
            "recommendation_type": "collaborative",
            "has_user_events": "Yes",
            "rec_placement": "product-page",
            "dataset_id": "retail_dataset",
            "user_events_table": "user_events",
        }
        if overrides:
            defaults.update(overrides)
        yaml_lines = "\n".join(f'{k}: "{v}"' for k, v in defaults.items())
        spec = tmp_path / "design-spec.md"
        spec.write_text(f"---\n{yaml_lines}\n---\n")
        return str(spec)

    @patch("_shared.setup_utils.subprocess.run")
    def test_collaborative_with_events_runs_ingestion(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path)
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "sample-user-events.csv").write_text("event_id\n1\n")

        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = setup(spec, dry_run=True)
            assert result is True
        finally:
            os.chdir(old_cwd)

    @patch("_shared.setup_utils.subprocess.run")
    def test_content_based_skips_events(self, mock_run, tmp_path, capsys):
        spec = self._make_spec(tmp_path, {"recommendation_type": "content-based"})
        result = setup(spec, dry_run=True)
        assert result is True
        mock_run.assert_not_called()

    @patch("_shared.setup_utils.subprocess.run")
    def test_collaborative_no_events_warns(self, mock_run, tmp_path):
        spec = self._make_spec(tmp_path, {"has_user_events": "No"})
        result = setup(spec, dry_run=True)
        assert result is True
        mock_run.assert_not_called()

    def test_missing_project_id_exits(self, tmp_path):
        spec = tmp_path / "design-spec.md"
        spec.write_text('---\nrecommendation_type: "collaborative"\n---\n')
        with pytest.raises(SystemExit):
            setup(str(spec))
