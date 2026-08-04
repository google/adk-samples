# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``horizon.a2a.datapart.unwrap_a2a_datapart_text``.

ADK wraps untyped A2A ``DataPart`` payloads (no ADK metadata-type key) in a
``<a2a_datapart_json>...</a2a_datapart_json>`` ``text/plain`` inline_data blob.
Such blobs are opaque to the model, so we unwrap them back to readable
text before they reach it.
"""

from __future__ import annotations

from horizon.a2a.datapart import (
    A2A_DATA_PART_END_TAG,
    A2A_DATA_PART_START_TAG,
    unwrap_a2a_datapart_text,
)


def _wrap(json_body: bytes) -> bytes:
    return A2A_DATA_PART_START_TAG + json_body + A2A_DATA_PART_END_TAG


def test_unwraps_ui_action_to_readable_text() -> None:
    body = (
        b'{"data":{"version":"v0.9","action":{"name":"gws_write_decision",'
        b'"surfaceId":"surface-1"}},"kind":"data"}'
    )
    text = unwrap_a2a_datapart_text(_wrap(body))
    assert text is not None
    assert "gws_write_decision" in text
    assert "surface-1" in text
    assert "a2a_datapart_json" not in text


def test_returns_none_for_non_datapart_bytes() -> None:
    assert unwrap_a2a_datapart_text(b"plain bytes, no tag") is None


def test_returns_none_for_empty() -> None:
    assert unwrap_a2a_datapart_text(None) is None
    assert unwrap_a2a_datapart_text(b"") is None


def test_tolerates_malformed_json_inside_tags() -> None:
    # Garbage between the tags must not raise — degrade to None so callers
    # fall back to their generic handling.
    assert unwrap_a2a_datapart_text(_wrap(b"{not json")) is None
