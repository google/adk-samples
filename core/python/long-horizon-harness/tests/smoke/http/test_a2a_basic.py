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

"""``/a2a`` accepts ``message/stream``, returns SSE, terminates cleanly.

Smoke for the full FastAPI → A2A → ADK Runner → Gemini wiring. Asserts
on event-stream shape only; no natural-language content checks.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import requests


def _stream_a2a(
    base_url: str, *, user_id: str, text: str, timeout: int = 120
) -> list[dict[str, Any]]:
    """POST ``message/stream`` and collect parsed SSE events."""
    payload = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/stream",
        "params": {
            "message": {
                "kind": "message",
                "messageId": uuid.uuid4().hex,
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
                "metadata": {"lha_user_id": user_id},
            }
        },
    }
    response = requests.post(
        f"{base_url}/a2a",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        stream=True,
        timeout=timeout,
    )
    assert response.status_code == 200, (
        f"/a2a returned {response.status_code}: {response.text[:500]}"
    )

    events: list[dict[str, Any]] = []
    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def test_a2a_stream_terminates_with_final_event(
    smoke_http_server: str, smoke_http_user_id: str, unique_marker: str
) -> None:
    events = _stream_a2a(
        smoke_http_server,
        user_id=smoke_http_user_id,
        text=(
            "Use the `terminal` tool to run exactly this command: "
            f"echo http-marker-{unique_marker}"
        ),
    )
    assert events, "no SSE events received from /a2a"

    final_seen = False
    for event in events:
        result = event.get("result") or {}
        if result.get("final") is True:
            final_seen = True
            break
        status = result.get("status") or {}
        if status.get("state") in {"completed", "failed", "canceled"}:
            final_seen = True
            break

    assert final_seen, (
        "expected stream to surface a terminal status/final flag; "
        f"got {len(events)} events without one"
    )
