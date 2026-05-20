"""Unit tests for setup_tryon.py — VTO + Gemini image generation setup."""

from unittest.mock import MagicMock, patch

import pytest

import setup_tryon
from setup_tryon import (
    ALL_MODELS,
    GEMINI_IMAGE_MODELS,
    SAFETY_LEVELS,
    VTO_MODEL,
    create_bucket_if_needed,
    load_config,
    resolve_model_id,
    setup,
    verify_gemini_image_api,
)


# ── Model resolution ─────────────────────────────────────────────────────────

class TestResolveModel:
    def test_vto_label_resolves_to_dedicated_model(self):
        assert resolve_model_id("vto") == "virtual-try-on-001"

    def test_gemini_labels_resolve_to_full_ids(self):
        assert resolve_model_id("flash") == "gemini-2.5-flash-image"
        assert resolve_model_id("pro") == "gemini-2.5-pro-image"

    def test_full_id_passes_through(self):
        assert resolve_model_id("gemini-2.5-flash-image") == "gemini-2.5-flash-image"
        assert resolve_model_id("virtual-try-on-001") == "virtual-try-on-001"

    def test_unknown_falls_back_to_vto(self):
        # vto is the default fallback (not flash) since it's purpose-built
        assert resolve_model_id("nonsense-model") == VTO_MODEL


# ── Bucket creation ──────────────────────────────────────────────────────────

class TestCreateBucket:
    def test_skips_when_bucket_exists(self):
        with patch.object(setup_tryon, "storage") as mock_storage:
            mock_client = mock_storage.Client.return_value
            mock_client.get_bucket.return_value = MagicMock()
            create_bucket_if_needed("project", "bucket-name")
            mock_client.create_bucket.assert_not_called()

    def test_creates_when_missing(self):
        with patch.object(setup_tryon, "storage") as mock_storage:
            mock_client = mock_storage.Client.return_value
            mock_client.get_bucket.side_effect = Exception("404")
            create_bucket_if_needed("project", "bucket-name")
            mock_client.create_bucket.assert_called_once()

    def test_applies_auto_delete_when_requested(self):
        with patch.object(setup_tryon, "storage") as mock_storage:
            mock_client = mock_storage.Client.return_value
            mock_client.get_bucket.side_effect = Exception("404")
            mock_bucket = MagicMock()
            mock_client.create_bucket.return_value = mock_bucket
            create_bucket_if_needed("project", "user-photos", auto_delete_days=1)
            mock_bucket.add_lifecycle_delete_rule.assert_called_once_with(age=1)


# ── End-to-end setup ─────────────────────────────────────────────────────────

class TestSetup:
    PROJECT = "test-project"
    LOCATION = "us-central1"
    OUTPUT = "test-project-tryon-output"
    UPLOAD = "test-project-tryon-uploads"

    def test_creates_both_buckets_and_verifies_api(self):
        with (
            patch.object(setup_tryon, "create_bucket_if_needed") as mock_create,
            patch.object(setup_tryon, "verify_gemini_image_api", return_value=True) as mock_verify,
        ):
            setup(self.PROJECT, self.LOCATION, self.OUTPUT, self.UPLOAD,
                  VTO_MODEL, "block_most")

            assert mock_create.call_count == 2
            mock_verify.assert_called_once()

    def test_upload_bucket_gets_24h_auto_delete(self):
        """User photo bucket must auto-delete after 24h for privacy."""
        with (
            patch.object(setup_tryon, "create_bucket_if_needed") as mock_create,
            patch.object(setup_tryon, "verify_gemini_image_api", return_value=True),
        ):
            setup(self.PROJECT, self.LOCATION, self.OUTPUT, self.UPLOAD,
                  VTO_MODEL, "block_most")

            # Find the upload-bucket call
            upload_call = next(c for c in mock_create.call_args_list
                               if c.args[1] == self.UPLOAD)
            assert upload_call.kwargs.get("auto_delete_days") == 1

    def test_continues_when_api_unavailable(self):
        """API check failure logs a warning but doesn't crash setup."""
        with (
            patch.object(setup_tryon, "create_bucket_if_needed"),
            patch.object(setup_tryon, "verify_gemini_image_api", return_value=False),
        ):
            # Should not raise
            setup(self.PROJECT, self.LOCATION, self.OUTPUT, self.UPLOAD,
                  VTO_MODEL, "block_most")


# ── Config loading ───────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_parses_yaml_frontmatter(self, sample_design_spec):
        cfg = load_config(sample_design_spec)
        assert cfg["gcp_project_id"] == "test-project"
        assert cfg["gemini_image_model"] == "flash"
        assert cfg["safety_level"] == "block_most"

    def test_returns_empty_for_missing_file(self):
        assert load_config("/does/not/exist.md") == {}


# ── Model and safety constants ───────────────────────────────────────────────

class TestConstants:
    def test_vto_is_dedicated_model(self):
        assert VTO_MODEL == "virtual-try-on-001"

    def test_all_gemini_tiers_defined(self):
        assert set(GEMINI_IMAGE_MODELS.keys()) == {"flash", "pro"}

    def test_all_models_includes_vto_and_gemini(self):
        assert set(ALL_MODELS.keys()) == {"vto", "flash", "pro"}
        assert ALL_MODELS["vto"] == "virtual-try-on-001"

    def test_all_safety_levels_defined(self):
        assert set(SAFETY_LEVELS.keys()) == {"block_most", "block_some", "block_few"}
