from unittest.mock import MagicMock, patch

from google.adk.tools import ToolContext

from youtube_analyst.tools import (
    init_or_get_youtube_client,
    store_youtube_api_key,
)


@patch("youtube_analyst.tools.config")
def test_init_or_get_youtube_client_missing_key(mock_config):
    """Test that init_or_get_youtube_client returns None and error message when key is missing."""
    mock_config.YOUTUBE_API_KEY = ""
    mock_context = MagicMock(spec=ToolContext)
    mock_context.state = {}

    client, error = init_or_get_youtube_client(mock_context)

    assert client is None
    assert "SYSTEM REJECTION: Missing YouTube API Key" in error


@patch("youtube_analyst.tools.config")
def test_init_or_get_youtube_client_env_fallback(mock_config):
    """Test that init_or_get_youtube_client falls back to environment key."""
    mock_config.YOUTUBE_API_KEY = "env_key_123"
    mock_context = MagicMock(spec=ToolContext)
    mock_context.state = {}

    # We patch 'build' to avoid actual API calls
    with patch("youtube_analyst.tools.build") as mock_build:
        client, error = init_or_get_youtube_client(mock_context)
        assert error is None
        mock_build.assert_called_with("youtube", "v3", developerKey="env_key_123")


def test_store_and_get_api_key():
    """Test storing a key and then attempting to initialize the client."""
    mock_context = MagicMock(spec=ToolContext)
    mock_context.state = {}

    test_key = "test_api_key_123"

    # Store the key
    result = store_youtube_api_key(test_key, mock_context)
    assert "successfully" in result
    assert mock_context.state["youtube_api_key"] == test_key

    # After storing, init_or_get_youtube_client should try to use the session key
    with patch("youtube_analyst.tools.build") as mock_build:
        client, error = init_or_get_youtube_client(mock_context)
        assert error is None
        mock_build.assert_called_with("youtube", "v3", developerKey=test_key)
