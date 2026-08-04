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

"""Outbound A2A push to the a2a-gateway ``/push`` endpoint.

Internal helper — NOT registered as a model-callable tool. The scheduler
tick endpoint (`POST /scheduler/tick`) calls it for every due reminder.

Payload shape mirrors the gateway's recently-updated inbound contract:
a JSON-RPC 2.0 ``message/send`` envelope wrapping an A2A ``Message``
with a single TextPart and two metadata fields tagging the destination
channel + recipient_id. The gateway translates that into a Telegram /
Slack / etc. send on the configured channel.

In prototype mode (``LHA_GATEWAY_URL`` unset) we log the payload to
stdout and return a no-op envelope so developers can exercise the
scheduler tick path without standing up a gateway.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GATEWAY_URL_ENV = "LHA_GATEWAY_URL"
_PUSH_PATH = "/push"
_REQUEST_TIMEOUT_S = 10.0


def _fetch_id_token(audience: str) -> str:
    """Fetch a Google-issued ID token for the given audience.

    Wrapped in its own function so unit tests can monkeypatch it without
    touching ``google.auth``.
    """
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token

    return id_token.fetch_id_token(Request(), audience)


def _build_payload(
    *, channel: str, recipient_id: str, text: str
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/send",
        "params": {
            "message": {
                "messageId": uuid.uuid4().hex,
                "role": "agent",
                "parts": [{"kind": "text", "text": text}],
                "metadata": {
                    "gateway/channel": channel,
                    "gateway/recipient_id": recipient_id,
                },
            }
        },
    }


async def push_to_user(
    *,
    user_id: str,
    app_name: str,
    channel: str,
    recipient_id: str,
    text: str,
) -> dict[str, Any]:
    """Push ``text`` to the user out-of-band via the a2a-gateway."""
    payload = _build_payload(
        channel=channel, recipient_id=recipient_id, text=text
    )

    gateway_url = os.environ.get(GATEWAY_URL_ENV, "").strip()
    if not gateway_url:
        print(
            "[push_to_user] LHA_GATEWAY_URL unset — would push:\n"
            f"  user_id={user_id} app_name={app_name} channel={channel} "
            f"recipient_id={recipient_id}\n"
            f"  payload={json.dumps(payload)}"
        )
        return {"delivered": False, "fallback": "stdout"}

    push_url = gateway_url.rstrip("/") + _PUSH_PATH
    headers = {"content-type": "application/json"}

    if push_url.startswith("https://"):
        try:
            token = _fetch_id_token(gateway_url.rstrip("/"))
        except Exception as exc:
            logger.exception("push_to_user: failed to mint ID token")
            return {"delivered": False, "error": f"id_token: {exc}"}
        headers["authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as client:
            response = await client.post(
                push_url, json=payload, headers=headers
            )
    except Exception as exc:
        logger.exception("push_to_user: gateway POST failed")
        return {"delivered": False, "error": str(exc)}

    if response.status_code >= 400:
        body = response.text[:512]
        logger.error(
            "push_to_user: gateway returned %s: %s", response.status_code, body
        )
        return {
            "delivered": False,
            "error": f"gateway {response.status_code}: {body}",
        }

    return {"delivered": True, "status_code": response.status_code}


__all__ = ["GATEWAY_URL_ENV", "push_to_user"]
