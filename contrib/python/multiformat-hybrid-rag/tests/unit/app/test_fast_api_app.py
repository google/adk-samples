import pytest
from pydantic import ValidationError

from app.config import MAX_TOP_K
from app.fast_api_app import SearchRequest


class TestSearchRequestValidation:
    """/api/search is a public trust boundary: top_k fans out into a Vector
    Search request per retriever and then into the Gemini prompt.
    """

    def test_defaults(self):
        req = SearchRequest(query="q")
        assert req.top_k == 10
        assert req.generative_answer is True

    def test_accepts_the_upper_bound(self):
        assert SearchRequest(query="q", top_k=MAX_TOP_K).top_k == MAX_TOP_K

    @pytest.mark.parametrize("bad", [MAX_TOP_K + 1, 10**9])
    def test_rejects_values_above_the_bound(self, bad):
        with pytest.raises(ValidationError):
            SearchRequest(query="q", top_k=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_non_positive(self, bad):
        with pytest.raises(ValidationError):
            SearchRequest(query="q", top_k=bad)
