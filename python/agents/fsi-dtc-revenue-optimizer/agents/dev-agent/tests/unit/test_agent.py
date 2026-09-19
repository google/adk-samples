import pytest
from unittest.mock import patch
from app.agent import send_google_chat_message

@pytest.mark.asyncio
async def test_send_google_chat_message_skip():
    with patch('app.agent.SKIP_CHAT', True):
        result = await send_google_chat_message("Test message")
        assert result == "ℹ️ Chat notification skipped because SKIP_CHAT is enabled."

@pytest.mark.asyncio
async def test_send_google_chat_message_no_webhook():
    with patch('app.agent.SKIP_CHAT', False):
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: "" if key == "GOOGLE_CHAT_WEBHOOK_URL" else None
            result = await send_google_chat_message("Test message")
            assert "Error: GOOGLE_CHAT_WEBHOOK_URL environment variable is not set." in result
