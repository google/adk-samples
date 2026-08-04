from unittest.mock import MagicMock, patch

from economic_research.sub_agents.tools.search_skill import web_search_skill


def test_web_search_skill_match():
    """Test web search skill for a known mock query using mocked requests."""
    with patch("requests.post") as mock_post:
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Found match for Austin",
                    "snippet": "Raleigh electricity rates",
                }
            ]
        }
        mock_post.return_value = mock_response

        # We set env var to bypass the check
        with patch("os.getenv", return_value="fake_key"):
            query = "Austin vs Raleigh"
            result = web_search_skill(query)
            assert "Results" in result
            assert "Austin" in result


def test_web_search_skill_generic():
    """Test web search skill for a generic query using mocked requests."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic": [
                {"title": "Detroit info", "snippet": "Unemployment stats"}
            ]
        }
        mock_post.return_value = mock_response

        with patch("os.getenv", return_value="fake_key"):
            query = "Detroit unemployment"
            result = web_search_skill(query)
            assert "Results" in result
            assert "Detroit" in result
