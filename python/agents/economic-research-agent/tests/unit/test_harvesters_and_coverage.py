"""Unit tests for Evolved Serper and EIA API Harvesters, and expanded tools coverage.
Created autonomously by AlphaEvolve.
"""

import json
import os
from unittest.mock import patch, MagicMock

import pytest
import us

from economic_research.tools.dynamic_search_harvester import (
    harvest_real_estate_roi,
    harvest_climate_risk,
    harvest_logistics_efficiency,
    harvest_cultural_amenities,
    harvest_regional_incentives,
    execute_serper_search,
    harvest_semantic_schema
)
from economic_research.tools.real_estate_skill import get_real_estate_roi
from economic_research.tools.climate_resilience_skill import get_climate_risk_index
from economic_research.tools.lifestyle_logistics_incentives_skills import (
    get_logistics_efficiency,
    get_cultural_amenity_score,
    get_regional_tax_incentives
)
from economic_research.tools.utility_logistics_skill import get_industrial_infrastructure_stats


@pytest.fixture
def mock_genai_client():
    with patch("google.genai.Client") as MockClient:
        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "Avg Lease (PSF)": "$40.00",
            "Vacancy Rate": "10.0%",
            "Overall Risk Rating": "Very High",
            "Primary Hazard (Heat)": "High",
            "Primary Hazard (Flood)": "Low",
            "Intermodal Hub Access": "Tier 1",
            "Shipping Cost Index (Lower=Better)": "95",
            "Transit Reliability Rate": "90%",
            "Walkability Score (0-100)": "75",
            "Amenity/Cultural Density": "High",
            "Safety Rating (FBI UCR)": "Very High",
            "Top Incentive Program": "Mock State Credit",
            "Estimated Subsidy Yield": "High Yield",
            "Statutory Corporate Credits": "Mock Credit A, B"
        })
        mock_instance.models.generate_content.return_value = mock_response
        yield MockClient


@pytest.fixture
def mock_serper_search():
    with patch("economic_research.tools.dynamic_search_harvester.execute_serper_search") as MockSerper:
        MockSerper.return_value = '{"organic": [{"snippet": "Mocked Serper Google Search Result"}]}'
        yield MockSerper


def test_execute_serper_search_no_key():
    with patch.dict(os.environ, {"SERPER_API_KEY": ""}, clear=False):
        result = execute_serper_search("Test Query")
        assert result == "{}"


def test_harvest_semantic_schema_fallback(mock_genai_client):
    # If serper returns empty, should immediately trigger fallbacks
    with patch("economic_research.tools.dynamic_search_harvester.execute_serper_search", return_value="{}"):
        fallbacks = {"Avg Lease (PSF)": "$12.00", "Vacancy Rate": "15.0%"}
        res = harvest_semantic_schema(
            "Query", "Instruction", ["Avg Lease (PSF)", "Vacancy Rate"], fallbacks
        )
        assert res == fallbacks


def test_harvest_real_estate_roi(mock_serper_search, mock_genai_client):
    res = harvest_real_estate_roi("Columbus, OH", property_type="Office")
    assert res["City"] == "Columbus"
    assert res["Property Type"] == "Office"
    assert res["Avg Lease (PSF)"] == "$40.00"
    assert "Source" in res


def test_harvest_climate_risk(mock_serper_search, mock_genai_client):
    res = harvest_climate_risk("Boise, ID")
    assert res["City"] == "Boise"
    assert res["Overall Risk Rating"] == "Very High"
    assert "Source" in res


def test_harvest_logistics_efficiency(mock_serper_search, mock_genai_client):
    res = harvest_logistics_efficiency("Scranton, PA")
    assert res["City"] == "Scranton"
    assert res["Intermodal Hub Access"] == "Tier 1"
    assert "Source" in res


def test_harvest_cultural_amenities(mock_serper_search, mock_genai_client):
    res = harvest_cultural_amenities("Des Moines, IA")
    assert res["City"] == "Des Moines"
    assert res["Walkability Score (0-100)"] == "75"
    assert "Source" in res


def test_harvest_regional_incentives(mock_serper_search, mock_genai_client):
    res = harvest_regional_incentives("Ohio")
    assert res["State"] == "Ohio"
    assert res["Top Incentive Program"] == "Mock State Credit"
    assert "Source" in res


def test_get_real_estate_roi_adapter(mock_serper_search, mock_genai_client):
    raw = get_real_estate_roi(["Austin, TX"], property_type="Industrial")
    data = json.loads(raw)
    assert len(data) == 1
    assert data[0]["City"] == "Austin"
    assert data[0]["Property Type"] == "Industrial"


def test_get_climate_risk_index_adapter(mock_serper_search, mock_genai_client):
    raw = get_climate_risk_index(["Miami, FL"])
    data = json.loads(raw)
    assert len(data) == 1
    assert data[0]["City"] == "Miami"


def test_get_logistics_efficiency_adapter(mock_serper_search, mock_genai_client):
    raw = get_logistics_efficiency(["Raleigh, NC"])
    data = json.loads(raw)
    assert len(data) == 1
    assert data[0]["City"] == "Raleigh"


def test_get_cultural_amenity_score_adapter(mock_serper_search, mock_genai_client):
    raw = get_cultural_amenity_score(["Boulder, CO"])
    data = json.loads(raw)
    assert len(data) == 1
    assert data[0]["City"] == "Boulder"


def test_get_regional_tax_incentives_adapter(mock_serper_search, mock_genai_client):
    raw = get_regional_tax_incentives(["Texas"])
    data = json.loads(raw)
    assert len(data) == 1
    assert data[0]["State"] == "Texas"


def test_get_industrial_infrastructure_stats():
    with patch("economic_research.tools.eia_skill.fetch_state_electricity_rates") as MockEIA:
        MockEIA.return_value = json.dumps([{
            "State": "TX",
            "Sector": "Industrial",
            "Avg Price (cents/kWh)": "8.50",
            "Period": "2024-03",
            "Source": "Mock"
        }])
        
        raw = get_industrial_infrastructure_stats(["Texas"])
        data = json.loads(raw)
        assert len(data) == 1
        assert data[0]["State"] == "Texas"
        assert data[0]["Industrial Elec (kWh)"] == "$0.085"
        assert "EIA Unified API Live" in data[0]["Source"]
