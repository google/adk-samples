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

"""``push_to_user`` — outbound A2A push to the a2a-gateway ``/push``.

Two execution modes:

* ``LHA_GATEWAY_URL`` unset → log the would-be payload to stdout and
  return ``{"delivered": False, "fallback": "stdout"}``. This is the
  prototype path so the scheduler tick endpoint works without a gateway.
* ``LHA_GATEWAY_URL`` set → POST a JSON-RPC ``message/send`` envelope
  carrying a TextPart + ``gateway/channel`` + ``gateway/recipient_id``
  metadata so the gateway can fan out to Telegram / Slack / etc.

Auth: when the URL is HTTPS, fetch a Google ID token via
``google.oauth2.id_token.fetch_id_token`` and put it in the Authorization
header (matches grocery-agent's outbound push). Skipped in dev / http.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

pytestmark = pytest.mark.asyncio


# =============================================================================
# Stdout fallback when GATEWAY_URL unset
# =============================================================================


class TestStdoutFallback:
    async def test_no_gateway_url_logs_to_stdout(self, monkeypatch, capsys):
        from horizon.tools.push_to_user import push_to_user

        monkeypatch.delenv("LHA_GATEWAY_URL", raising=False)
        result = await push_to_user(
            user_id="u1",
            app_name="lha",
            channel="cli",
            recipient_id="u1",
            text="reminder: buy milk",
        )
        assert result["delivered"] is False
        assert result["fallback"] == "stdout"
        captured = capsys.readouterr()
        assert "buy milk" in captured.out

    async def test_empty_gateway_url_also_falls_back(self, monkeypatch):
        from horizon.tools.push_to_user import push_to_user

        monkeypatch.setenv("LHA_GATEWAY_URL", "")
        result = await push_to_user(
            user_id="u1",
            app_name="lha",
            channel="cli",
            recipient_id="u1",
            text="hi",
        )
        assert result["delivered"] is False


# =============================================================================
# JSON-RPC payload shape over HTTP
# =============================================================================


class TestPayloadShape:
    @respx.mock
    async def test_posts_to_push_path_with_jsonrpc(self, monkeypatch):
        from horizon.tools.push_to_user import push_to_user

        monkeypatch.setenv("LHA_GATEWAY_URL", "http://gateway.local:8080")
        route = respx.post("http://gateway.local:8080/push").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        result = await push_to_user(
            user_id="u1",
            app_name="lha",
            channel="telegram",
            recipient_id="chat-42",
            text="ping",
        )
        assert result["delivered"] is True
        assert route.called
        body = json.loads(route.calls[0].request.content)
        assert body["jsonrpc"] == "2.0"
        assert body["method"] == "message/send"
        assert "id" in body
        message = body["params"]["message"]
        assert message["role"] == "agent"
        assert "messageId" in message
        parts = message["parts"]
        assert len(parts) == 1
        assert parts[0]["kind"] == "text"
        assert parts[0]["text"] == "ping"
        metadata = message["metadata"]
        assert metadata["gateway/channel"] == "telegram"
        assert metadata["gateway/recipient_id"] == "chat-42"

    @respx.mock
    async def test_http_error_returns_delivered_false(self, monkeypatch):
        from horizon.tools.push_to_user import push_to_user

        monkeypatch.setenv("LHA_GATEWAY_URL", "http://gateway.local:8080")
        respx.post("http://gateway.local:8080/push").mock(
            return_value=httpx.Response(500, json={"error": "boom"})
        )
        result = await push_to_user(
            user_id="u1",
            app_name="lha",
            channel="cli",
            recipient_id="u1",
            text="x",
        )
        assert result["delivered"] is False
        assert "error" in result

    @respx.mock
    async def test_trailing_slash_on_url_is_handled(self, monkeypatch):
        from horizon.tools.push_to_user import push_to_user

        monkeypatch.setenv("LHA_GATEWAY_URL", "http://gateway.local:8080/")
        route = respx.post("http://gateway.local:8080/push").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        result = await push_to_user(
            user_id="u1",
            app_name="lha",
            channel="cli",
            recipient_id="u1",
            text="x",
        )
        assert result["delivered"] is True
        assert route.called


# =============================================================================
# OIDC auth on HTTPS Cloud Run URLs
# =============================================================================


class TestAuth:
    @respx.mock
    async def test_https_url_adds_bearer_token(self, monkeypatch):
        from horizon.tools import push_to_user as mod

        monkeypatch.setenv("LHA_GATEWAY_URL", "https://gateway-abc.a.run.app")
        monkeypatch.setattr(
            mod, "_fetch_id_token", lambda audience: "fake-id-token"
        )
        route = respx.post("https://gateway-abc.a.run.app/push").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        result = await mod.push_to_user(
            user_id="u1",
            app_name="lha",
            channel="cli",
            recipient_id="u1",
            text="x",
        )
        assert result["delivered"] is True
        assert route.called
        auth = route.calls[0].request.headers.get("authorization", "")
        assert auth == "Bearer fake-id-token"

    @respx.mock
    async def test_http_url_skips_token_fetch(self, monkeypatch):
        from horizon.tools import push_to_user as mod

        monkeypatch.setenv("LHA_GATEWAY_URL", "http://gateway.local:8080")
        called = {"n": 0}

        def boom(audience):
            called["n"] += 1
            return "should-not-be-used"

        monkeypatch.setattr(mod, "_fetch_id_token", boom)
        respx.post("http://gateway.local:8080/push").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        result = await mod.push_to_user(
            user_id="u1",
            app_name="lha",
            channel="cli",
            recipient_id="u1",
            text="x",
        )
        assert result["delivered"] is True
        assert called["n"] == 0
