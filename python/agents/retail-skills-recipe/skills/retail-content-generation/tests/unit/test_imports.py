"""Import validation tests for retail-content-generation scripts.

Verifies that all scripts can be imported without errors and that
key functions/constants are accessible after import.
"""

import pytest


class TestScriptImports:
    """Verify all scripts import successfully with mocked GCP modules."""

    def test_import_generate_content(self):
        import generate_content
        assert hasattr(generate_content, "build_prompt")
        assert hasattr(generate_content, "generate_for_product")
        assert hasattr(generate_content, "save_to_file")
        assert hasattr(generate_content, "load_config")
        assert hasattr(generate_content, "CONTENT_TYPES")
        assert hasattr(generate_content, "DEFAULT_MODEL")

    def test_import_export_content(self):
        import export_content
        assert hasattr(export_content, "export")
        assert hasattr(export_content, "load_config")

    def test_generate_content_constants(self):
        import generate_content as gc
        assert isinstance(gc.CONTENT_TYPES, list)
        assert len(gc.CONTENT_TYPES) == 5
        assert gc.DEFAULT_MODEL == "gemini-2.5-flash"
        assert isinstance(gc.DEFAULT_TONE, str)
