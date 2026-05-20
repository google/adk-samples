"""Unit tests for generate_content.py.

Covers three retail use cases:
  1. Electronics store - product descriptions and SEO titles
  2. Fashion store - marketing copy with luxury brand voice
  3. Grocery store - translations and A/B variants

Plus cross-cutting tests for prompt building, file output, and config loading.
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

import generate_content as gc


# ===================================================================
# Constants
# ===================================================================

class TestConstants:
    """Verify module-level constants."""

    def test_content_types_contains_all_five(self):
        assert gc.CONTENT_TYPES == [
            "description", "seo_title", "meta_description",
            "marketing_copy", "translation",
        ]

    def test_default_model(self):
        assert gc.DEFAULT_MODEL == "gemini-2.5-flash"

    def test_default_tone(self):
        assert gc.DEFAULT_TONE == "Professional and informative"


# ===================================================================
# Use Case 1: Electronics Store
# ===================================================================

class TestElectronicsDescriptions:
    """Electronics: product descriptions and SEO titles."""

    def test_description_prompt_includes_product_name(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[0]  # headphones
        prompt = gc.build_prompt(product, "description", electronics_config)
        assert "Wireless Noise-Cancelling Headphones" in prompt

    def test_description_prompt_includes_category(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", electronics_config)
        assert "Audio" in prompt

    def test_description_prompt_includes_price(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", electronics_config)
        assert "$179.99" in prompt

    def test_description_prompt_includes_rating(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", electronics_config)
        assert "4.6/5" in prompt

    def test_description_prompt_includes_tone(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", electronics_config)
        assert "Technical and enthusiastic" in prompt

    def test_description_prompt_includes_specs_section(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", electronics_config)
        assert "technical specifications" in prompt.lower()

    def test_description_prompt_includes_use_cases(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", electronics_config)
        assert "Who it is for" in prompt

    def test_description_prompt_excludes_specs_when_disabled(self, typed_electronics_products):
        config = {"include_specs": False, "include_use_cases": False}
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", config)
        assert "technical specifications" not in prompt.lower()
        assert "Who it is for" not in prompt

    def test_description_prompt_target_length(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", electronics_config)
        assert "medium" in prompt.lower()

    def test_seo_title_prompt_for_keyboard(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[2]  # keyboard
        prompt = gc.build_prompt(product, "seo_title", electronics_config)
        assert "SEO-optimized" in prompt
        assert "50-60 characters" in prompt
        assert "Mechanical Gaming Keyboard" in prompt

    def test_seo_title_excludes_price_by_default(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[2]
        prompt = gc.build_prompt(product, "seo_title", electronics_config)
        assert "Do NOT include the price" in prompt

    def test_seo_title_includes_price_when_configured(self, typed_electronics_products):
        config = {"include_price_in_seo": True}
        product = typed_electronics_products[2]
        prompt = gc.build_prompt(product, "seo_title", config)
        assert "Include the price" in prompt

    def test_seo_title_returns_only_title(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[2]
        prompt = gc.build_prompt(product, "seo_title", electronics_config)
        assert "Return ONLY the title" in prompt

    def test_generate_description_calls_gemini(self, typed_electronics_products, electronics_config, mock_gemini_client):
        product = typed_electronics_products[0]
        result = gc.generate_for_product(
            mock_gemini_client, "gemini-2.5-flash", product, "description", electronics_config,
        )
        mock_gemini_client.models.generate_content.assert_called_once()
        assert result == "Generated content goes here."

    def test_generate_description_uses_low_temperature(self, typed_electronics_products, electronics_config, mock_gemini_client):
        product = typed_electronics_products[0]
        gc.generate_for_product(
            mock_gemini_client, "gemini-2.5-flash", product, "description", electronics_config,
        )
        call_kwargs = mock_gemini_client.models.generate_content.call_args
        config_arg = call_kwargs.kwargs.get("config")
        assert config_arg.temperature == 0.3

    def test_product_without_rating_omits_rating_line(self, electronics_config):
        product = {
            "product_id": "elec-004",
            "name": "USB Cable",
            "price": 9.99,
            "category": "Cables",
            "brand": "CableCo",
            "description": "A simple USB cable",
        }
        prompt = gc.build_prompt(product, "description", electronics_config)
        assert "Rating:" not in prompt

    def test_all_three_electronics_products_generate_prompts(self, typed_electronics_products, electronics_config):
        for product in typed_electronics_products:
            prompt = gc.build_prompt(product, "description", electronics_config)
            assert product["name"] in prompt
            assert len(prompt) > 50


# ===================================================================
# Use Case 2: Fashion Store
# ===================================================================

class TestFashionMarketingCopy:
    """Fashion: marketing copy and meta descriptions with luxury brand voice."""

    def test_marketing_copy_prompt_format(self, fashion_product, fashion_config):
        prompt = gc.build_prompt(fashion_product, "marketing_copy", fashion_config)
        assert "marketing headline" in prompt.lower()
        assert "Headline:" in prompt
        assert "Subheadline:" in prompt

    def test_marketing_copy_includes_luxury_tone(self, fashion_product, fashion_config):
        prompt = gc.build_prompt(fashion_product, "marketing_copy", fashion_config)
        assert "Luxurious and aspirational" in prompt

    def test_marketing_copy_includes_brand_name(self, fashion_product, fashion_config):
        prompt = gc.build_prompt(fashion_product, "marketing_copy", fashion_config)
        assert "Brooks & Co" in prompt

    def test_marketing_copy_includes_always_include(self, fashion_product, fashion_config):
        prompt = gc.build_prompt(fashion_product, "marketing_copy", fashion_config)
        assert "craftsmanship, heritage" in prompt

    def test_marketing_copy_includes_never_use(self, fashion_product, fashion_config):
        prompt = gc.build_prompt(fashion_product, "marketing_copy", fashion_config)
        assert "cheap, discount, bargain" in prompt

    def test_marketing_copy_uses_high_temperature(self, fashion_product, fashion_config, mock_gemini_client):
        gc.generate_for_product(
            mock_gemini_client, "gemini-2.5-flash", fashion_product, "marketing_copy", fashion_config,
        )
        call_kwargs = mock_gemini_client.models.generate_content.call_args
        config_arg = call_kwargs.kwargs.get("config")
        assert config_arg.temperature == 0.7

    def test_meta_description_prompt(self, fashion_product, fashion_config):
        prompt = gc.build_prompt(fashion_product, "meta_description", fashion_config)
        assert "meta description" in prompt.lower()
        assert "150-160 characters" in prompt
        assert "call to action" in prompt.lower()

    def test_meta_description_includes_product_info(self, fashion_product, fashion_config):
        prompt = gc.build_prompt(fashion_product, "meta_description", fashion_config)
        assert "Cashmere Wrap Coat" in prompt
        assert "$895.0" in prompt

    def test_meta_description_uses_low_temperature(self, fashion_product, fashion_config, mock_gemini_client):
        gc.generate_for_product(
            mock_gemini_client, "gemini-2.5-flash", fashion_product, "meta_description", fashion_config,
        )
        call_kwargs = mock_gemini_client.models.generate_content.call_args
        config_arg = call_kwargs.kwargs.get("config")
        assert config_arg.temperature == 0.3

    def test_config_brand_name_overrides_product_brand(self, fashion_config):
        """When config has brand_name, it should be used instead of product brand."""
        product = {
            "product_id": "fash-002",
            "name": "Silk Scarf",
            "price": 250.00,
            "description": "Hand-printed silk scarf",
            "category": "Accessories",
            "brand": "OtherBrand",
        }
        prompt = gc.build_prompt(product, "marketing_copy", fashion_config)
        assert "Brooks & Co" in prompt

    def test_no_brand_name_in_config_uses_product_brand(self):
        product = {
            "product_id": "fash-003",
            "name": "Leather Belt",
            "price": 120.00,
            "description": "Genuine leather belt",
            "category": "Accessories",
            "brand": "BeltMaker",
        }
        config = {"brand_tone": "Classic"}
        prompt = gc.build_prompt(product, "marketing_copy", config)
        assert "BeltMaker" in prompt


# ===================================================================
# Use Case 3: Grocery Store
# ===================================================================

class TestGroceryTranslations:
    """Grocery: translations and A/B variant descriptions."""

    def test_translation_prompt_includes_target_language(self, grocery_product):
        config = {"target_language": "es"}
        prompt = gc.build_prompt(grocery_product, "translation", config)
        assert "Translate" in prompt
        assert "es" in prompt

    def test_translation_prompt_preserves_brand_names(self, grocery_product):
        config = {"target_language": "fr"}
        prompt = gc.build_prompt(grocery_product, "translation", config)
        assert "Do NOT translate brand names" in prompt

    def test_translation_prompt_includes_original_description(self, grocery_product):
        config = {"target_language": "es"}
        prompt = gc.build_prompt(grocery_product, "translation", config)
        assert "Cold-pressed single-origin organic olive oil" in prompt

    def test_translation_prompt_for_french(self, grocery_product):
        config = {"target_language": "fr"}
        prompt = gc.build_prompt(grocery_product, "translation", config)
        assert "fr" in prompt

    def test_translation_default_language_is_spanish(self, grocery_product):
        config = {}
        prompt = gc.build_prompt(grocery_product, "translation", config)
        assert "es" in prompt

    def test_translation_returns_only_translated_text(self, grocery_product):
        config = {"target_language": "es"}
        prompt = gc.build_prompt(grocery_product, "translation", config)
        assert "Return ONLY the translated text" in prompt

    def test_translation_uses_low_temperature(self, grocery_product, mock_gemini_client):
        config = {"target_language": "es"}
        gc.generate_for_product(
            mock_gemini_client, "gemini-2.5-flash", grocery_product, "translation", config,
        )
        call_kwargs = mock_gemini_client.models.generate_content.call_args
        config_arg = call_kwargs.kwargs.get("config")
        assert config_arg.temperature == 0.3

    def test_description_prompt_with_grocery_config(self, grocery_product, grocery_config):
        prompt = gc.build_prompt(grocery_product, "description", grocery_config)
        assert "Warm and wholesome" in prompt
        assert "organic, fresh, natural" in prompt

    def test_description_short_length(self, grocery_product, grocery_config):
        prompt = gc.build_prompt(grocery_product, "description", grocery_config)
        assert "short" in prompt.lower()

    def test_generate_strips_whitespace(self, grocery_product, mock_gemini_client):
        mock_gemini_client.models.generate_content.return_value.text = "  translated text  \n"
        result = gc.generate_for_product(
            mock_gemini_client, "gemini-2.5-flash", grocery_product, "translation", {},
        )
        assert result == "translated text"


# ===================================================================
# Cross-cutting: Prompt Building
# ===================================================================

class TestPromptBuilding:
    """Cross-cutting prompt construction tests."""

    def test_default_tone_when_not_configured(self, typed_electronics_products):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", {})
        assert "Professional and informative" in prompt

    def test_unknown_content_type_fallback(self, typed_electronics_products):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "unknown_type", {})
        assert "Describe this product" in prompt
        assert product["name"] in prompt

    def test_empty_config_uses_defaults(self, typed_electronics_products):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", {})
        assert "Medium (100-150 words)" not in prompt  # split logic uses default
        assert "Professional and informative" in prompt

    def test_always_include_omitted_when_empty(self, typed_electronics_products):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", {})
        assert "Always include:" not in prompt

    def test_never_use_omitted_when_empty(self, typed_electronics_products):
        product = typed_electronics_products[0]
        prompt = gc.build_prompt(product, "description", {})
        assert "Never use" not in prompt

    def test_product_without_image_still_works(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[1]  # docking station, no image_url
        prompt = gc.build_prompt(product, "description", electronics_config)
        assert "USB-C Docking Station" in prompt

    def test_all_content_types_return_nonempty_prompts(self, typed_electronics_products, electronics_config):
        product = typed_electronics_products[0]
        for content_type in gc.CONTENT_TYPES:
            prompt = gc.build_prompt(product, content_type, electronics_config)
            assert len(prompt) > 20, f"Prompt too short for {content_type}"


# ===================================================================
# Save to File
# ===================================================================

class TestSaveToFile:
    """File output tests."""

    def test_save_json_creates_valid_json(self, tmp_path):
        results = [
            {"product_id": "elec-001", "content_type": "description", "content": "Great headphones"},
            {"product_id": "elec-002", "content_type": "description", "content": "Awesome dock"},
        ]
        output = str(tmp_path / "output.json")
        gc.save_to_file(results, output, "json")

        loaded = json.loads(Path(output).read_text())
        assert len(loaded) == 2
        assert loaded[0]["product_id"] == "elec-001"
        assert loaded[1]["content"] == "Awesome dock"

    def test_save_json_is_indented(self, tmp_path):
        results = [{"product_id": "p1", "content": "text"}]
        output = str(tmp_path / "output.json")
        gc.save_to_file(results, output, "json")

        text = Path(output).read_text()
        assert "\n" in text  # indented JSON has newlines

    def test_save_csv_creates_valid_csv(self, tmp_path):
        results = [
            {"product_id": "elec-001", "content_type": "description", "content": "Great headphones"},
            {"product_id": "elec-002", "content_type": "seo_title", "content": "Best Dock 2025"},
        ]
        output = str(tmp_path / "output.csv")
        gc.save_to_file(results, output, "csv")

        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["product_id"] == "elec-001"
        assert rows[1]["content_type"] == "seo_title"

    def test_save_csv_has_header(self, tmp_path):
        results = [{"product_id": "p1", "content_type": "description", "content": "text"}]
        output = str(tmp_path / "output.csv")
        gc.save_to_file(results, output, "csv")

        with open(output) as f:
            first_line = f.readline().strip()
        assert "product_id" in first_line
        assert "content_type" in first_line
        assert "content" in first_line

    def test_save_json_with_translation_fields(self, tmp_path):
        results = [
            {
                "product_id": "groc-001",
                "content_type": "translation",
                "content": "Aceite de oliva virgen extra organico",
                "language": "es",
                "model": "gemini-2.5-flash",
                "variant": 1,
            },
        ]
        output = str(tmp_path / "translations.json")
        gc.save_to_file(results, output, "json")

        loaded = json.loads(Path(output).read_text())
        assert loaded[0]["language"] == "es"
        assert loaded[0]["variant"] == 1


# ===================================================================
# Config Loading
# ===================================================================

class TestConfigLoading:
    """Config loading tests."""

    def test_load_yaml_config(self, sample_config_yaml):
        config = gc.load_config(sample_config_yaml)
        assert config["gcp_project_id"] == "test-project-123"
        assert config["brand_tone"] == "Professional and informative"

    def test_load_design_spec_frontmatter(self, sample_design_spec):
        config = gc.load_config(sample_design_spec)
        assert config["gcp_project_id"] == "test-project-123"
        assert config["brand_tone"] == "Luxurious and aspirational"
        assert config["brand_name"] == "Brooks & Co"
        assert config["always_include"] == "craftsmanship, heritage"

    def test_load_nonexistent_file_returns_empty(self):
        config = gc.load_config("/nonexistent/path/config.yaml")
        assert config == {}

    def test_load_empty_yaml_returns_empty(self, tmp_path):
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        config = gc.load_config(str(empty_file))
        assert config == {}

    def test_load_design_spec_ignores_markdown_body(self, sample_design_spec):
        config = gc.load_config(sample_design_spec)
        # Should not contain markdown content
        assert "Design Spec" not in str(config)
        assert "test design spec" not in str(config)
