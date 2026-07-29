import os
import pytest
from economic_research.agent import set_session_api_key

def test_set_session_api_key_valid():
    key_name = "FRED_API_KEY"
    key_value = "test_value"
    
    # Save old value to restore later
    old_value = os.environ.get(key_name)
    
    result = set_session_api_key(key_name, key_value)
    
    assert "Successfully set" in result
    assert os.environ.get(key_name) == key_value
    
    # Restore or clean up
    if old_value:
        os.environ[key_name] = old_value
    else:
        del os.environ[key_name]

def test_set_session_api_key_invalid():
    key_name = "INVALID_KEY"
    key_value = "test_value"
    
    result = set_session_api_key(key_name, key_value)
    
    assert "ERROR" in result
    assert os.environ.get(key_name) is None
