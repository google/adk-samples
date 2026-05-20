"""Tests for content generation setup.py - config-driven pipeline orchestrator."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.setup import load_config, setup


class TestLoadConfig:
    def test_loads_content_config(self, tmp_path):
        spec = tmp_path / "design-spec.md"
        spec.write_text('---\ngcp_project_id: "proj"\ncontent_type: "description"\n---\n')
        cfg = load_config(str(spec))
        assert cfg["content_type"] == "description"


class TestSetupContentGeneration:
    def _make_spec(self, tmp_path, overrides=None):
        defaults = {
            "gcp_project_id": "test-project",
            "content_type": "description",
            "brand_tone": "Professional",
            "ab_variants": "1",
            "gemini_model": "gemini-2.5-flash",
            "batch_size": "50",
            "output_format": "BigQuery table",
            "dataset_id": "products_dataset",
        }
        if overrides:
            defaults.update(overrides)
        yaml_lines = "\n".join(f'{k}: "{v}"' for k, v in defaults.items())
        # Add target_languages
        langs = overrides.get("target_languages", []) if overrides else []
        if langs:
            langs_yaml = "\n".join(f'  - "{l}"' for l in langs)
            langs_section = f"\ntarget_languages:\n{langs_yaml}"
        else:
            langs_section = "\ntarget_languages: []"
        spec = tmp_path / "design-spec.md"
        spec.write_text(f"---\n{yaml_lines}{langs_section}\n---\n")
        return str(spec)

    @patch("_shared.setup_utils.subprocess.run")
    def test_generates_description_content(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path)
        result = setup(spec, dry_run=True)
        assert result is True

    @patch("_shared.setup_utils.subprocess.run")
    def test_all_type_runs_multiple(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path, {"content_type": "all"})
        setup(spec, dry_run=True)
        # "all" should call generate for description, seo_title, meta_description, marketing_copy

    @patch("_shared.setup_utils.subprocess.run")
    def test_with_translations(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path, {"target_languages": ["es", "fr"]})
        setup(spec, dry_run=True)

    @patch("_shared.setup_utils.subprocess.run")
    def test_multiple_variants(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path, {"ab_variants": "3"})
        result = setup(spec, dry_run=True)
        assert result is True

    def test_missing_project_id_exits(self, tmp_path):
        spec = tmp_path / "design-spec.md"
        spec.write_text('---\ncontent_type: "description"\n---\n')
        with pytest.raises(SystemExit):
            setup(str(spec))
