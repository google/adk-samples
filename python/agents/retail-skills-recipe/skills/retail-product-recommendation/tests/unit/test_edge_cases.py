"""Edge case tests for retail-product-recommendation scripts.

Covers duplicate events, future timestamps, unusual metadata,
and boundary conditions.
"""

import json

import pytest

import ingest_user_events as mod


def _make_event(**overrides):
    """Return a minimal valid event dict, with optional overrides."""
    base = {
        "event_id": "evt-edge",
        "user_id": "user-1",
        "product_id": "prod-1",
        "event_type": "detail-page-view",
        "timestamp": "2025-01-15T10:00:00Z",
    }
    base.update(overrides)
    return base


class TestDuplicateEvents:
    """Handling of duplicate event_ids."""

    def test_duplicate_event_ids_both_pass_validation(self):
        """Duplicate event_ids should both pass individual validation."""
        events = [
            _make_event(event_id="dup-001"),
            _make_event(event_id="dup-001"),
        ]
        for i, event in enumerate(events):
            errors = mod.validate_event(event, i + 1)
            assert errors == []

    def test_validate_and_convert_preserves_duplicates(self):
        """_validate_and_convert does not deduplicate (that's BQ's job)."""
        events = [
            _make_event(event_id="dup-001"),
            _make_event(event_id="dup-001"),
        ]
        result = mod._validate_and_convert(events)
        assert len(result) == 2


class TestTimestampEdgeCases:
    """Edge cases for event timestamps."""

    def test_future_timestamp_passes_validation(self):
        """Future timestamps pass validation (no range check in current code)."""
        event = _make_event(timestamp="2030-12-31T23:59:59Z")
        errors = mod.validate_event(event, 1)
        assert errors == []

    def test_very_old_timestamp(self):
        """Very old timestamps should still pass validation."""
        event = _make_event(timestamp="2000-01-01T00:00:00Z")
        errors = mod.validate_event(event, 1)
        assert errors == []

    def test_timestamp_with_timezone_offset(self):
        """Timestamps with timezone offsets should be accepted."""
        event = _make_event(timestamp="2025-01-15T10:00:00+05:30")
        errors = mod.validate_event(event, 1)
        assert errors == []


class TestMetadataEdgeCases:
    """Edge cases for event metadata."""

    def test_large_metadata(self):
        """Very large metadata dict should serialize."""
        large_meta = {f"key_{i}": f"value_{i}" for i in range(100)}
        events = [_make_event(event_metadata=large_meta)]
        result = mod._validate_and_convert(events)
        assert len(result) == 1
        parsed = json.loads(result[0]["event_metadata"])
        assert len(parsed) == 100

    def test_nested_metadata(self):
        """Nested metadata should serialize to JSON."""
        nested = {"cart": {"items": [{"id": "p1", "qty": 2}], "total": 29.98}}
        events = [_make_event(event_metadata=nested)]
        result = mod._validate_and_convert(events)
        parsed = json.loads(result[0]["event_metadata"])
        assert parsed["cart"]["items"][0]["id"] == "p1"

    def test_metadata_with_unicode(self):
        """Metadata with unicode should serialize correctly."""
        events = [_make_event(event_metadata={"query": "cafe latte"})]
        result = mod._validate_and_convert(events)
        parsed = json.loads(result[0]["event_metadata"])
        assert "cafe" in parsed["query"]


class TestLargeBatches:
    """Test behavior with large event lists."""

    def test_validate_large_batch(self):
        """500 events should all validate."""
        events = [
            _make_event(event_id=f"evt-{i:04d}")
            for i in range(500)
        ]
        result = mod._validate_and_convert(events)
        assert len(result) == 500

    def test_mixed_valid_invalid_large_batch(self):
        """In a large batch, only valid events should survive."""
        events = []
        for i in range(100):
            if i % 10 == 0:
                events.append(_make_event(event_id=""))  # invalid
            else:
                events.append(_make_event(event_id=f"evt-{i}"))
        result = mod._validate_and_convert(events)
        assert len(result) == 90
