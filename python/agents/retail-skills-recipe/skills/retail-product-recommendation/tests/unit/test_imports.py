"""Import validation tests for retail-product-recommendation scripts.

Verifies that all scripts can be imported without errors and that
key functions/constants are accessible after import.
"""

import pytest


class TestScriptImports:
    """Verify all scripts import successfully with mocked GCP modules."""

    def test_import_ingest_user_events(self):
        import ingest_user_events
        assert hasattr(ingest_user_events, "validate_event")
        assert hasattr(ingest_user_events, "load_from_csv")
        assert hasattr(ingest_user_events, "load_config")
        assert hasattr(ingest_user_events, "VALID_EVENT_TYPES")
        assert hasattr(ingest_user_events, "REQUIRED_FIELDS")

    def test_ingest_user_events_constants(self):
        import ingest_user_events as mod
        assert isinstance(mod.VALID_EVENT_TYPES, set)
        assert len(mod.VALID_EVENT_TYPES) == 6
        assert "purchase" in mod.VALID_EVENT_TYPES
        assert isinstance(mod.REQUIRED_FIELDS, list)
        assert len(mod.REQUIRED_FIELDS) == 5
