"""Unit tests for tryon_agent.py -- the ADK tool wrapper."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make app/ importable
APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP_DIR.parent))

from app.tryon_agent import try_on_product, SUPPORTED_CATEGORIES


class TestSupportedCategories:
    def test_includes_core_categories(self):
        assert "clothing" in SUPPORTED_CATEGORIES
        assert "eyewear" in SUPPORTED_CATEGORIES
        assert "footwear" in SUPPORTED_CATEGORIES

    def test_is_a_set(self):
        assert isinstance(SUPPORTED_CATEGORIES, set)


class TestTryOnProductMissingConfig:
    """When env vars are not set, try_on_product should return an error."""

    def test_missing_project_id(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.setenv("TRYON_OUTPUT_BUCKET", "out")
        monkeypatch.setenv("TRYON_UPLOAD_BUCKET", "up")

        result = try_on_product("prod-1", "gs://bucket/photo.jpg")
        assert result["status"] == "error"
        assert "not configured" in result["error"]

    def test_missing_output_bucket(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
        monkeypatch.delenv("TRYON_OUTPUT_BUCKET", raising=False)
        monkeypatch.setenv("TRYON_UPLOAD_BUCKET", "up")

        result = try_on_product("prod-1", "gs://bucket/photo.jpg")
        assert result["status"] == "error"

    def test_missing_upload_bucket(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
        monkeypatch.setenv("TRYON_OUTPUT_BUCKET", "out")
        monkeypatch.delenv("TRYON_UPLOAD_BUCKET", raising=False)

        result = try_on_product("prod-1", "gs://bucket/photo.jpg")
        assert result["status"] == "error"

    def test_all_missing(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("TRYON_OUTPUT_BUCKET", raising=False)
        monkeypatch.delenv("TRYON_UPLOAD_BUCKET", raising=False)

        result = try_on_product("prod-1", "gs://bucket/photo.jpg")
        assert result["status"] == "error"
        assert "setup_tryon.py" in result["error"]


class TestTryOnProductSuccess:
    """When env vars are set and generate_tryon succeeds."""

    @patch("app.tryon_agent.generate_tryon")
    def test_success_returns_expected_keys(self, mock_generate, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
        monkeypatch.setenv("TRYON_OUTPUT_BUCKET", "out")
        monkeypatch.setenv("TRYON_UPLOAD_BUCKET", "up")

        mock_generate.return_value = {
            "output_uri": "gs://out/tryon-output/prod-1/abc.jpg",
            "all_variations": ["gs://out/tryon-output/prod-1/abc.jpg"],
            "model_used": "virtual-try-on-001",
            "variations_generated": 1,
        }

        result = try_on_product("prod-1", "gs://bucket/photo.jpg", "denim jacket")

        assert result["status"] == "success"
        assert result["output_uri"] == "gs://out/tryon-output/prod-1/abc.jpg"
        assert result["model_used"] == "virtual-try-on-001"
        assert result["variations"] == 1
        assert len(result["all_variations"]) == 1

    @patch("app.tryon_agent.generate_tryon")
    def test_passes_env_vars_to_generate_tryon(self, mock_generate, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-proj")
        monkeypatch.setenv("TRYON_OUTPUT_BUCKET", "my-out")
        monkeypatch.setenv("TRYON_UPLOAD_BUCKET", "my-up")
        monkeypatch.setenv("GEMINI_IMAGE_MODEL", "pro")
        monkeypatch.setenv("TRYON_SAFETY_LEVEL", "block_few")

        mock_generate.return_value = {
            "output_uri": "gs://my-out/x.jpg",
            "all_variations": ["gs://my-out/x.jpg"],
            "model_used": "gemini-2.5-pro-image",
            "variations_generated": 1,
        }

        try_on_product("prod-1", "gs://bucket/photo.jpg")

        mock_generate.assert_called_once_with(
            product_id="prod-1",
            user_photo_uri="gs://bucket/photo.jpg",
            project_id="my-proj",
            output_bucket="my-out",
            upload_bucket="my-up",
            model_label_or_id="pro",
            safety_level="block_few",
            product_description="",
        )


class TestTryOnProductError:
    """When generate_tryon raises an exception."""

    @patch("app.tryon_agent.generate_tryon")
    def test_exception_returns_error_status(self, mock_generate, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
        monkeypatch.setenv("TRYON_OUTPUT_BUCKET", "out")
        monkeypatch.setenv("TRYON_UPLOAD_BUCKET", "up")

        mock_generate.side_effect = RuntimeError("SAFETY filter blocked")

        result = try_on_product("prod-1", "gs://bucket/photo.jpg")

        assert result["status"] == "error"
        assert "SAFETY filter blocked" in result["error"]

    @patch("app.tryon_agent.generate_tryon")
    def test_generic_exception_captured(self, mock_generate, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
        monkeypatch.setenv("TRYON_OUTPUT_BUCKET", "out")
        monkeypatch.setenv("TRYON_UPLOAD_BUCKET", "up")

        mock_generate.side_effect = ValueError("bad input")

        result = try_on_product("prod-1", "gs://bucket/photo.jpg")

        assert result["status"] == "error"
        assert "bad input" in result["error"]
