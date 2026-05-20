"""Unit tests for tryon_processor.py -- model resolution, routing, and generate_tryon."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Make app/ importable
APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP_DIR.parent))

from app.tryon_processor import (
    MODELS,
    SAFETY_LEVELS,
    DEFAULT_LABEL,
    DEFAULT_SAFETY_LEVEL,
    resolve_model,
    is_vto_model,
    _make_client,
    is_product_cutout,
    generate_tryon,
)


# -- Model resolution ---------------------------------------------------------

class TestResolveModel:
    def test_vto_label(self):
        assert resolve_model("vto") == "virtual-try-on-001"

    def test_flash_label(self):
        assert resolve_model("flash") == "gemini-2.5-flash-image"

    def test_pro_label(self):
        assert resolve_model("pro") == "gemini-2.5-pro-image"

    def test_full_id_passthrough(self):
        assert resolve_model("gemini-2.5-flash-image") == "gemini-2.5-flash-image"
        assert resolve_model("virtual-try-on-001") == "virtual-try-on-001"

    def test_empty_string_defaults_to_vto(self):
        assert resolve_model("") == "virtual-try-on-001"

    def test_unknown_label_passes_through(self):
        assert resolve_model("custom-model-v2") == "custom-model-v2"


class TestIsVtoModel:
    def test_vto_model_id(self):
        assert is_vto_model("virtual-try-on-001") is True

    def test_gemini_flash_is_not_vto(self):
        assert is_vto_model("gemini-2.5-flash-image") is False

    def test_gemini_pro_is_not_vto(self):
        assert is_vto_model("gemini-2.5-pro-image") is False

    def test_future_vto_model(self):
        assert is_vto_model("virtual-try-on-002") is True

    def test_arbitrary_model_is_not_vto(self):
        assert is_vto_model("some-other-model") is False


# -- Constants -----------------------------------------------------------------

class TestConstants:
    def test_all_model_labels_defined(self):
        assert set(MODELS.keys()) == {"vto", "flash", "pro"}

    def test_vto_uses_recontext_api(self):
        assert MODELS["vto"]["api"] == "recontext_image"

    def test_gemini_models_use_generate_content(self):
        assert MODELS["flash"]["api"] == "generate_content"
        assert MODELS["pro"]["api"] == "generate_content"

    def test_all_safety_levels_defined(self):
        assert set(SAFETY_LEVELS.keys()) == {"block_most", "block_some", "block_few"}

    def test_defaults(self):
        assert DEFAULT_LABEL == "vto"
        assert DEFAULT_SAFETY_LEVEL == "block_most"


# -- Pre-flight classifier ----------------------------------------------------

class TestIsProductCutout:
    @patch("app.tryon_processor._make_client")
    def test_returns_true_for_product(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "product"
        mock_client.models.generate_content.return_value = mock_response

        assert is_product_cutout(b"fake-image-bytes", "test-project") is True

    @patch("app.tryon_processor._make_client")
    def test_returns_false_for_person(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "person"
        mock_client.models.generate_content.return_value = mock_response

        assert is_product_cutout(b"fake-image-bytes", "test-project") is False

    @patch("app.tryon_processor._make_client")
    def test_handles_empty_response(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response

        assert is_product_cutout(b"fake-image-bytes", "test-project") is False


# -- generate_tryon (mocked end-to-end) ----------------------------------------

class TestGenerateTryon:
    PROJECT = "test-project"
    OUTPUT_BUCKET = "test-output"
    UPLOAD_BUCKET = "test-uploads"
    PRODUCT_ID = "jacket-001"
    USER_PHOTO = "gs://test-uploads/photo.jpg"

    def _mock_gcs(self):
        """Create a mock GCS client that returns fake image bytes."""
        mock_gcs = MagicMock()
        mock_bucket = MagicMock()
        mock_gcs.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = b"fake-image-data"
        mock_bucket.blob.return_value = mock_blob
        return mock_gcs

    @patch("app.tryon_processor.is_product_cutout", return_value=False)
    @patch("app.tryon_processor._make_client")
    @patch("app.tryon_processor.storage")
    def test_vto_path_calls_recontext_image(self, mock_storage_mod, mock_client_factory, mock_cutout):
        mock_gcs = self._mock_gcs()
        mock_storage_mod.Client.return_value = mock_gcs

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_gen = MagicMock()
        mock_gen.image.image_bytes = b"output-image"
        mock_client.models.recontext_image.return_value = MagicMock(
            generated_images=[mock_gen]
        )

        result = generate_tryon(
            product_id=self.PRODUCT_ID,
            user_photo_uri=self.USER_PHOTO,
            project_id=self.PROJECT,
            output_bucket=self.OUTPUT_BUCKET,
            upload_bucket=self.UPLOAD_BUCKET,
            model_label_or_id="vto",
        )

        assert result["model_used"] == "virtual-try-on-001"
        assert result["variations_generated"] == 1
        mock_client.models.recontext_image.assert_called_once()

    @patch("app.tryon_processor.is_product_cutout", return_value=False)
    @patch("app.tryon_processor._make_client")
    @patch("app.tryon_processor.storage")
    def test_gemini_path_calls_generate_content(self, mock_storage_mod, mock_client_factory, mock_cutout):
        mock_gcs = self._mock_gcs()
        mock_storage_mod.Client.return_value = mock_gcs

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        mock_part = MagicMock()
        mock_part.inline_data.mime_type = "image/jpeg"
        mock_part.inline_data.data = b"gemini-output"
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        result = generate_tryon(
            product_id=self.PRODUCT_ID,
            user_photo_uri=self.USER_PHOTO,
            project_id=self.PROJECT,
            output_bucket=self.OUTPUT_BUCKET,
            upload_bucket=self.UPLOAD_BUCKET,
            model_label_or_id="flash",
            num_variations=1,
        )

        assert result["model_used"] == "gemini-2.5-flash-image"
        assert result["variations_generated"] == 1
        mock_client.models.generate_content.assert_called_once()

    @patch("app.tryon_processor.is_product_cutout", return_value=False)
    @patch("app.tryon_processor._make_client")
    @patch("app.tryon_processor.storage")
    def test_no_output_raises_runtime_error(self, mock_storage_mod, mock_client_factory, mock_cutout):
        mock_gcs = self._mock_gcs()
        mock_storage_mod.Client.return_value = mock_gcs

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_client.models.recontext_image.return_value = MagicMock(generated_images=[])

        with pytest.raises(RuntimeError, match="All .* try-on variations failed"):
            generate_tryon(
                product_id=self.PRODUCT_ID,
                user_photo_uri=self.USER_PHOTO,
                project_id=self.PROJECT,
                output_bucket=self.OUTPUT_BUCKET,
                upload_bucket=self.UPLOAD_BUCKET,
                model_label_or_id="vto",
            )

    @patch("app.tryon_processor.is_product_cutout", return_value=True)
    @patch("app.tryon_processor._make_client")
    @patch("app.tryon_processor.storage")
    def test_cutout_relaxes_safety_for_gemini(self, mock_storage_mod, mock_client_factory, mock_cutout):
        mock_gcs = self._mock_gcs()
        mock_storage_mod.Client.return_value = mock_gcs

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        mock_part = MagicMock()
        mock_part.inline_data.mime_type = "image/jpeg"
        mock_part.inline_data.data = b"output"
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        result = generate_tryon(
            product_id=self.PRODUCT_ID,
            user_photo_uri=self.USER_PHOTO,
            project_id=self.PROJECT,
            output_bucket=self.OUTPUT_BUCKET,
            upload_bucket=self.UPLOAD_BUCKET,
            model_label_or_id="flash",
            safety_level="block_most",
            num_variations=1,
        )

        assert result["safety_level"] == "block_some"

    @patch("app.tryon_processor.is_product_cutout", return_value=True)
    @patch("app.tryon_processor._make_client")
    @patch("app.tryon_processor.storage")
    def test_cutout_does_not_relax_safety_for_vto(self, mock_storage_mod, mock_client_factory, mock_cutout):
        mock_gcs = self._mock_gcs()
        mock_storage_mod.Client.return_value = mock_gcs

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_gen = MagicMock()
        mock_gen.image.image_bytes = b"output"
        mock_client.models.recontext_image.return_value = MagicMock(
            generated_images=[mock_gen]
        )

        result = generate_tryon(
            product_id=self.PRODUCT_ID,
            user_photo_uri=self.USER_PHOTO,
            project_id=self.PROJECT,
            output_bucket=self.OUTPUT_BUCKET,
            upload_bucket=self.UPLOAD_BUCKET,
            model_label_or_id="vto",
            safety_level="block_most",
            num_variations=1,
        )

        assert result["safety_level"] == "block_most"

    @patch("app.tryon_processor.is_product_cutout", return_value=False)
    @patch("app.tryon_processor._make_client")
    @patch("app.tryon_processor.storage")
    def test_base64_user_photo(self, mock_storage_mod, mock_client_factory, mock_cutout):
        """Verify base64-encoded user photos are decoded correctly."""
        import base64
        mock_gcs = self._mock_gcs()
        mock_storage_mod.Client.return_value = mock_gcs

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_gen = MagicMock()
        mock_gen.image.image_bytes = b"output"
        mock_client.models.recontext_image.return_value = MagicMock(
            generated_images=[mock_gen]
        )

        b64_photo = base64.b64encode(b"fake-photo").decode()
        result = generate_tryon(
            product_id=self.PRODUCT_ID,
            user_photo_uri=b64_photo,
            project_id=self.PROJECT,
            output_bucket=self.OUTPUT_BUCKET,
            upload_bucket=self.UPLOAD_BUCKET,
            model_label_or_id="vto",
            num_variations=1,
        )

        assert result["model_used"] == "virtual-try-on-001"
        assert result["variations_generated"] == 1

    @patch("app.tryon_processor.is_product_cutout", return_value=False)
    @patch("app.tryon_processor._make_client")
    @patch("app.tryon_processor.storage")
    def test_output_uris_use_correct_bucket(self, mock_storage_mod, mock_client_factory, mock_cutout):
        mock_gcs = self._mock_gcs()
        mock_storage_mod.Client.return_value = mock_gcs

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_gen = MagicMock()
        mock_gen.image.image_bytes = b"output"
        mock_client.models.recontext_image.return_value = MagicMock(
            generated_images=[mock_gen]
        )

        result = generate_tryon(
            product_id=self.PRODUCT_ID,
            user_photo_uri=self.USER_PHOTO,
            project_id=self.PROJECT,
            output_bucket=self.OUTPUT_BUCKET,
            upload_bucket=self.UPLOAD_BUCKET,
            model_label_or_id="vto",
            num_variations=1,
        )

        assert result["output_uri"].startswith(f"gs://{self.OUTPUT_BUCKET}/")
        assert all(u.startswith(f"gs://{self.OUTPUT_BUCKET}/") for u in result["all_variations"])
