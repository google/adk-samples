from youtube_analyst.tools import get_date_range, parse_timestamp_to_seconds


def test_parse_timestamp_to_seconds():
    """Test various timestamp formats."""
    assert parse_timestamp_to_seconds('322') == 322 # noqa: PLR2004
    assert parse_timestamp_to_seconds('05:22') == 322 # noqa: PLR2004
    assert parse_timestamp_to_seconds('5:22') == 322 # noqa: PLR2004
    assert parse_timestamp_to_seconds('1:05:22') == 3922 # noqa: PLR2004
    assert parse_timestamp_to_seconds('01:05:22') == 3922 # noqa: PLR2004
    assert parse_timestamp_to_seconds('invalid') == 0
    assert parse_timestamp_to_seconds('') == 0

def test_get_date_range_valid():
    """Test valid date range keys."""
    for span in ['week', 'month', '3month', 'year']:
        result = get_date_range(span)
        assert isinstance(result, str)
        assert len(result) > 0
        # Basic check for ISO format start
        assert result[0].isdigit()

def test_get_date_range_invalid():
    """Test invalid date range keys."""
    assert get_date_range('day') == ""
    assert get_date_range('forever') == ""
    assert get_date_range('') == ""
