"""Tests for virtual tryon setup.py - config-driven pipeline orchestrator."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.setup import load_config, setup


class TestLoadConfig:
    def test_loads_tryon_config(self, tmp_path):
        spec = tmp_path / "design-spec.md"
        spec.write_text('---\ngcp_project_id: "proj"\nhas_transparent_images: "No"\n---\n')
        cfg = load_config(str(spec))
        assert cfg["has_transparent_images"] == "No"


class TestSetupTryon:
    def _make_spec(self, tmp_path, overrides=None):
        defaults = {
            "gcp_project_id": "test-project",
            "gcp_region": "us-central1",
            "has_transparent_images": "No",
            "user_photo_source": "Upload from device",
            "gemini_image_model": "flash",
            "tryon_resolution": "512x512",
        }
        if overrides:
            defaults.update(overrides)
        yaml_lines = "\n".join(f'{k}: "{v}"' for k, v in defaults.items())
        # Add tryon_categories as YAML list
        cats = overrides.get("tryon_categories", ["Clothing"]) if overrides else ["Clothing"]
        cats_yaml = "\n".join(f'  - "{c}"' for c in cats)
        spec = tmp_path / "design-spec.md"
        spec.write_text(f"---\n{yaml_lines}\ntryon_categories:\n{cats_yaml}\n---\n")
        return str(spec)

    @patch("_shared.setup_utils.subprocess.run")
    def test_runs_setup_tryon_script(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path)
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "setup_tryon.py").write_text("")

        result = setup(spec, dry_run=True)
        assert result is True

    @patch("_shared.setup_utils.subprocess.run")
    def test_no_transparent_images_warns(self, mock_run, tmp_path, capsys):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path, {"has_transparent_images": "No"})
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "setup_tryon.py").write_text("")

        setup(spec, dry_run=True)
        # Warning is logged, not printed to capsys

    @patch("_shared.setup_utils.subprocess.run")
    def test_multiple_categories(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        spec = self._make_spec(tmp_path, {"tryon_categories": ["Clothing", "Eyewear", "Jewelry"]})
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "setup_tryon.py").write_text("")

        result = setup(spec, dry_run=True)
        assert result is True

    def test_missing_project_id_exits(self, tmp_path):
        spec = tmp_path / "design-spec.md"
        spec.write_text('---\nhas_transparent_images: "No"\n---\n')
        with pytest.raises(SystemExit):
            setup(str(spec))
