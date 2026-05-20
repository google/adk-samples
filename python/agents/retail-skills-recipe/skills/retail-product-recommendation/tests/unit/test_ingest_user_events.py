"""Unit tests for ingest_user_events.py.

Organised by use-case:
  - E-commerce: browse-to-purchase funnel
  - Fashion: discovery, wishlist, and purchase
  - Grocery: repeat purchases and reviews
  - Invalid events: cross-cutting validation
  - Event type validation: all valid/invalid types
  - Config loading: YAML and design-spec.md
"""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import ingest_user_events as mod


# ===================================================================
# Helpers
# ===================================================================

def _make_event(**overrides):
    """Return a minimal valid event dict, with optional overrides."""
    base = {
        "event_id": "evt-test",
        "user_id": "user-1",
        "product_id": "prod-1",
        "event_type": "detail-page-view",
        "timestamp": "2025-01-15T10:00:00Z",
    }
    base.update(overrides)
    return base


# ===================================================================
# Use Case 1: E-commerce Browse & Purchase
# ===================================================================

class TestEcommerceEvents:
    """E-commerce: browse-to-purchase funnel events."""

    def test_load_ecommerce_csv(self, ecommerce_events_csv):
        """All 6 e-commerce events load successfully."""
        events = mod.load_from_csv(ecommerce_events_csv)
        assert len(events) == 6

    def test_ecommerce_event_types(self, ecommerce_events_csv):
        """E-commerce funnel includes view, add-to-cart, and purchase."""
        events = mod.load_from_csv(ecommerce_events_csv)
        types_found = {e["event_type"] for e in events}
        assert types_found == {"detail-page-view", "add-to-cart", "purchase"}

    def test_ecommerce_user_ids(self, ecommerce_events_csv):
        """Two distinct users in the e-commerce data."""
        events = mod.load_from_csv(ecommerce_events_csv)
        user_ids = {e["user_id"] for e in events}
        assert user_ids == {"user-100", "user-101"}

    def test_ecommerce_session_ids(self, ecommerce_events_csv):
        """Each user has their own session."""
        events = mod.load_from_csv(ecommerce_events_csv)
        sessions = {e.get("session_id") for e in events if e.get("session_id")}
        assert sessions == {"sess-001", "sess-002"}

    def test_ecommerce_metadata_serialised(self, ecommerce_events_csv):
        """Purchase event has JSON metadata with quantity."""
        events = mod.load_from_csv(ecommerce_events_csv)
        purchase = [e for e in events if e["event_id"] == "evt-003"][0]
        assert "event_metadata" in purchase
        parsed = json.loads(purchase["event_metadata"])
        assert parsed["quantity"] == 1

    def test_ecommerce_validate_and_convert(self, ecommerce_events):
        """_validate_and_convert succeeds on valid e-commerce rows."""
        result = mod._validate_and_convert(ecommerce_events)
        assert len(result) == 6
        for event in result:
            for field in mod.REQUIRED_FIELDS:
                assert field in event

    def test_ecommerce_events_without_metadata(self, ecommerce_events_csv):
        """Events without metadata do not include event_metadata key."""
        events = mod.load_from_csv(ecommerce_events_csv)
        no_meta = [e for e in events if e["event_id"] == "evt-004"]
        assert len(no_meta) == 1
        assert "event_metadata" not in no_meta[0]


# ===================================================================
# Use Case 2: Fashion Discovery
# ===================================================================

class TestFashionEvents:
    """Fashion: discovery, wishlist, and purchase events."""

    def test_load_fashion_csv(self, fashion_events_csv):
        """All 6 fashion events load successfully."""
        events = mod.load_from_csv(fashion_events_csv)
        assert len(events) == 6

    def test_fashion_event_types(self, fashion_events_csv):
        """Fashion flow includes search, view, wishlist, cart, purchase."""
        events = mod.load_from_csv(fashion_events_csv)
        types_found = {e["event_type"] for e in events}
        expected = {"search", "detail-page-view", "add-to-wishlist", "add-to-cart", "purchase"}
        assert types_found == expected

    def test_fashion_single_user(self, fashion_events_csv):
        """All fashion events belong to user-200."""
        events = mod.load_from_csv(fashion_events_csv)
        assert all(e["user_id"] == "user-200" for e in events)

    def test_fashion_single_session(self, fashion_events_csv):
        """All fashion events are in session sess-010."""
        events = mod.load_from_csv(fashion_events_csv)
        assert all(e.get("session_id") == "sess-010" for e in events)

    def test_fashion_no_metadata(self, fashion_events_csv):
        """Fashion events have no event_metadata."""
        events = mod.load_from_csv(fashion_events_csv)
        assert all("event_metadata" not in e for e in events)

    def test_fashion_validate_and_convert(self, fashion_events):
        """_validate_and_convert succeeds on valid fashion rows."""
        result = mod._validate_and_convert(fashion_events)
        assert len(result) == 6


# ===================================================================
# Use Case 3: Grocery Repeat Customer
# ===================================================================

class TestGroceryEvents:
    """Grocery: repeat purchases and reviews."""

    def test_load_grocery_csv(self, grocery_events_csv):
        """All 6 grocery events load successfully."""
        events = mod.load_from_csv(grocery_events_csv)
        assert len(events) == 6

    def test_grocery_event_types(self, grocery_events_csv):
        """Grocery flow includes purchase, review, and view."""
        events = mod.load_from_csv(grocery_events_csv)
        types_found = {e["event_type"] for e in events}
        assert types_found == {"purchase", "review", "detail-page-view"}

    def test_grocery_single_user(self, grocery_events_csv):
        """All grocery events belong to user-300."""
        events = mod.load_from_csv(grocery_events_csv)
        assert all(e["user_id"] == "user-300" for e in events)

    def test_grocery_multiple_sessions(self, grocery_events_csv):
        """Grocery user has 3 sessions across multiple days."""
        events = mod.load_from_csv(grocery_events_csv)
        sessions = {e.get("session_id") for e in events if e.get("session_id")}
        assert sessions == {"sess-020", "sess-021", "sess-022"}

    def test_grocery_repeat_product_purchase(self, grocery_events_csv):
        """groc-001 appears in both purchase and review events."""
        events = mod.load_from_csv(grocery_events_csv)
        groc001 = [e for e in events if e["product_id"] == "groc-001"]
        assert len(groc001) == 2
        types_found = {e["event_type"] for e in groc001}
        assert types_found == {"purchase", "review"}

    def test_grocery_validate_and_convert(self, grocery_events):
        """_validate_and_convert succeeds on valid grocery rows."""
        result = mod._validate_and_convert(grocery_events)
        assert len(result) == 6


# ===================================================================
# Cross-cutting: Invalid Event Validation
# ===================================================================

class TestInvalidEvents:
    """Cross-cutting: invalid event validation."""

    def test_missing_event_id(self):
        """Missing event_id produces a validation error."""
        event = _make_event(event_id="")
        errors = mod.validate_event(event, 1)
        assert any("event_id" in e for e in errors)

    def test_missing_user_id(self):
        """Missing user_id produces a validation error."""
        event = _make_event(user_id="")
        errors = mod.validate_event(event, 1)
        assert any("user_id" in e for e in errors)

    def test_missing_product_id(self):
        """Missing product_id produces a validation error."""
        event = _make_event(product_id="")
        errors = mod.validate_event(event, 1)
        assert any("product_id" in e for e in errors)

    def test_missing_event_type(self):
        """Missing event_type produces a validation error."""
        event = _make_event(event_type="")
        errors = mod.validate_event(event, 1)
        assert any("event_type" in e for e in errors)

    def test_missing_timestamp(self):
        """Missing timestamp produces a validation error."""
        event = _make_event(timestamp="")
        errors = mod.validate_event(event, 1)
        assert any("timestamp" in e for e in errors)

    def test_invalid_event_type(self):
        """Invalid event_type produces a validation error."""
        event = _make_event(event_type="invalid-event-type")
        errors = mod.validate_event(event, 1)
        assert any("invalid event_type" in e for e in errors)

    def test_valid_event_no_errors(self):
        """A fully valid event produces no validation errors."""
        event = _make_event()
        errors = mod.validate_event(event, 1)
        assert errors == []

    def test_missing_field_key(self):
        """Event dict without a required key produces a validation error."""
        event = {"user_id": "u1", "product_id": "p1", "event_type": "purchase", "timestamp": "2025-01-01T00:00:00Z"}
        errors = mod.validate_event(event, 1)
        assert any("event_id" in e for e in errors)

    def test_load_invalid_csv_skips_bad_rows(self, invalid_events_csv):
        """Only the valid row survives when loading the invalid CSV."""
        events = mod.load_from_csv(invalid_events_csv)
        assert len(events) == 1
        assert events[0]["event_id"] == "evt-good"

    def test_validate_and_convert_skips_invalid(self, invalid_events):
        """_validate_and_convert filters out invalid rows."""
        result = mod._validate_and_convert(invalid_events)
        assert len(result) == 1
        assert result[0]["event_id"] == "evt-good"

    def test_row_number_in_error_message(self):
        """Error messages include the row number."""
        event = _make_event(event_id="")
        errors = mod.validate_event(event, 42)
        assert any("Row 42" in e for e in errors)

    def test_multiple_missing_fields(self):
        """An event missing multiple fields reports all of them."""
        event = {"event_type": "purchase", "timestamp": "2025-01-01T00:00:00Z"}
        errors = mod.validate_event(event, 1)
        assert len(errors) >= 3  # event_id, user_id, product_id


# ===================================================================
# Event Type Validation
# ===================================================================

class TestEventTypeValidation:
    """Verify all valid/invalid event types."""

    @pytest.mark.parametrize("event_type", [
        "detail-page-view",
        "add-to-cart",
        "purchase",
        "search",
        "add-to-wishlist",
        "review",
    ])
    def test_valid_event_type_accepted(self, event_type):
        """Each of the 6 valid event types passes validation."""
        event = _make_event(event_type=event_type)
        errors = mod.validate_event(event, 1)
        assert errors == []

    @pytest.mark.parametrize("event_type", [
        "click",
        "view",
        "buy",
        "page-view",
        "PURCHASE",
        "add_to_cart",
    ])
    def test_invalid_event_type_rejected(self, event_type):
        """Various invalid event types are caught."""
        event = _make_event(event_type=event_type)
        errors = mod.validate_event(event, 1)
        assert any("invalid event_type" in e for e in errors)

    def test_valid_event_types_constant(self):
        """VALID_EVENT_TYPES contains exactly 6 types."""
        assert len(mod.VALID_EVENT_TYPES) == 6
        expected = {"detail-page-view", "add-to-cart", "purchase", "search", "add-to-wishlist", "review"}
        assert mod.VALID_EVENT_TYPES == expected

    def test_required_fields_constant(self):
        """REQUIRED_FIELDS contains exactly 5 fields."""
        assert len(mod.REQUIRED_FIELDS) == 5
        expected = ["event_id", "user_id", "product_id", "event_type", "timestamp"]
        assert mod.REQUIRED_FIELDS == expected


# ===================================================================
# Event Metadata Handling
# ===================================================================

class TestEventMetadata:
    """Test event_metadata serialisation behaviour."""

    def test_string_metadata_preserved(self):
        """String metadata is kept as-is."""
        events = [_make_event(event_metadata='{"quantity": 2}')]
        result = mod._validate_and_convert(events)
        assert result[0]["event_metadata"] == '{"quantity": 2}'

    def test_dict_metadata_serialised(self):
        """Dict metadata is JSON-serialised to a string."""
        events = [_make_event(event_metadata={"quantity": 2})]
        result = mod._validate_and_convert(events)
        parsed = json.loads(result[0]["event_metadata"])
        assert parsed["quantity"] == 2

    def test_empty_metadata_omitted(self):
        """Empty string metadata is not included in output."""
        events = [_make_event(event_metadata="")]
        result = mod._validate_and_convert(events)
        assert "event_metadata" not in result[0]

    def test_no_metadata_field_omitted(self):
        """Event without event_metadata key omits it from output."""
        events = [_make_event()]
        result = mod._validate_and_convert(events)
        assert "event_metadata" not in result[0]

    def test_session_id_included_when_present(self):
        """session_id is carried through when present."""
        events = [_make_event(session_id="sess-abc")]
        result = mod._validate_and_convert(events)
        assert result[0]["session_id"] == "sess-abc"

    def test_session_id_omitted_when_empty(self):
        """Empty session_id is not included in output."""
        events = [_make_event(session_id="")]
        result = mod._validate_and_convert(events)
        assert "session_id" not in result[0]

    def test_session_id_omitted_when_absent(self):
        """Missing session_id key is not included in output."""
        events = [_make_event()]
        result = mod._validate_and_convert(events)
        assert "session_id" not in result[0]


# ===================================================================
# Config Loading
# ===================================================================

class TestConfigLoading:
    """Config loading tests."""

    def test_load_yaml_config(self, sample_config_yaml):
        """load_config reads a plain YAML file."""
        cfg = mod.load_config(sample_config_yaml)
        assert cfg["gcp_project_id"] == "test-project-123"
        assert cfg["gcp_region"] == "us-central1"

    def test_load_design_spec(self, sample_design_spec):
        """load_config reads YAML frontmatter from design-spec.md."""
        cfg = mod.load_config(sample_design_spec)
        assert cfg["gcp_project_id"] == "test-project-123"
        assert cfg["dataset_id"] == "products_dataset"

    def test_load_missing_config_returns_empty(self, tmp_path):
        """load_config returns empty dict for nonexistent file."""
        cfg = mod.load_config(str(tmp_path / "nonexistent.yaml"))
        assert cfg == {}

    def test_load_empty_yaml_returns_empty(self, tmp_path):
        """load_config returns empty dict for empty YAML file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        cfg = mod.load_config(str(empty_file))
        assert cfg == {}


# ===================================================================
# JSON / JSONL loading
# ===================================================================

class TestJsonLoading:
    """Test load_from_json with local files."""

    def test_load_json_array(self, tmp_path):
        """load_from_json handles a JSON array of events."""
        events = [_make_event(event_id=f"evt-{i}") for i in range(3)]
        json_file = tmp_path / "events.json"
        json_file.write_text(json.dumps(events))
        result = mod.load_from_json(str(json_file))
        assert len(result) == 3

    def test_load_json_object_with_events_key(self, tmp_path):
        """load_from_json handles {"events": [...]} format."""
        events = [_make_event(event_id=f"evt-{i}") for i in range(2)]
        json_file = tmp_path / "events.json"
        json_file.write_text(json.dumps({"events": events}))
        result = mod.load_from_json(str(json_file))
        assert len(result) == 2

    def test_load_jsonl(self, tmp_path):
        """load_from_json handles JSONL format."""
        events = [_make_event(event_id=f"evt-{i}") for i in range(4)]
        jsonl_file = tmp_path / "events.jsonl"
        jsonl_file.write_text("\n".join(json.dumps(e) for e in events))
        result = mod.load_from_json(str(jsonl_file))
        assert len(result) == 4

    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        """JSONL loader skips blank lines."""
        events = [_make_event(event_id=f"evt-{i}") for i in range(2)]
        jsonl_file = tmp_path / "events.jsonl"
        content = json.dumps(events[0]) + "\n\n" + json.dumps(events[1]) + "\n"
        jsonl_file.write_text(content)
        result = mod.load_from_json(str(jsonl_file))
        assert len(result) == 2

    def test_load_json_with_invalid_events(self, tmp_path):
        """Invalid events in JSON are filtered out."""
        events = [
            _make_event(event_id="good"),
            _make_event(event_id=""),  # invalid
        ]
        json_file = tmp_path / "events.json"
        json_file.write_text(json.dumps(events))
        result = mod.load_from_json(str(json_file))
        assert len(result) == 1
        assert result[0]["event_id"] == "good"
