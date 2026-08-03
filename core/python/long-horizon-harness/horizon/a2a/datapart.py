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

"""Recover ADK's opaque A2A DataPart blobs as readable text.

When an A2A ``DataPart`` carries no ADK metadata-type key (an untyped
client-sent payload), ADK's ``convert_a2a_part_to_genai_part`` wraps its JSON in a
``<a2a_datapart_json>...</a2a_datapart_json>`` ``text/plain`` ``inline_data``
blob. Some model adapters have no branch for such blobs and can error on them,
and even when tolerated the tagged blob is opaque to the model. These pure
helpers let the inbound converter turn that blob back into text the model can
actually read.

Tags mirror ``google.adk.a2a.converters.part_converter``; kept inline so this
module stays dependency-free (no A2A SDK import).
"""

from __future__ import annotations

import json

A2A_DATA_PART_START_TAG = b"<a2a_datapart_json>"
A2A_DATA_PART_END_TAG = b"</a2a_datapart_json>"


def unwrap_a2a_datapart_text(data: bytes | None) -> str | None:
    """Return readable text for a wrapped DataPart blob, else ``None``.

    ``None`` signals "not one of these blobs" so callers fall back to their
    own handling (placeholder substitution, pass-through, etc.).
    """
    if not data or not data.startswith(A2A_DATA_PART_START_TAG):
        return None
    body = data[len(A2A_DATA_PART_START_TAG) :]
    if body.endswith(A2A_DATA_PART_END_TAG):
        body = body[: -len(A2A_DATA_PART_END_TAG)]
    try:
        obj = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    payload = obj.get("data", obj) if isinstance(obj, dict) else obj
    return "[UI action] " + json.dumps(payload, ensure_ascii=False)


__all__ = [
    "A2A_DATA_PART_END_TAG",
    "A2A_DATA_PART_START_TAG",
    "unwrap_a2a_datapart_text",
]
