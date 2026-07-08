# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""Unit tests for Dynamic routing and self-measurement of economic primitives."""

import os
import json
import glob
import shutil
from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture(autouse=True)
def mock_keys(monkeypatch):
    """Mock all API keys in environment to prevent get_cloud_secret calling gcloud/metadata credentials."""
    allowed_keys = [
        "BEA_API_KEY", "FRED_API_KEY", "CENSUS_API_KEY", "EIA_API_KEY",
        "BLS_API_KEY", "HUD_API_KEY", "FEC_API_KEY", "NEWS_API_KEY",
        "SERPER_API_KEY", "CDC_APP_TOKEN", "OPENFDA_API_KEY"
    ]
    for key in allowed_keys:
        monkeypatch.setenv(key, f"mock_{key.lower()}")


@patch('google.adk.runners.InMemoryRunner.run')
def test_query_routing_and_primitives_low_complexity(mock_run):
    from economic_research.agent import export_agent
    
    log_dir = "/Users/enriq/.gemini/jetski/scratch/observability"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        
    def make_mock_response(text):
        mock_part = MagicMock()
        mock_part.text = text
        mock_res = MagicMock()
        mock_res.content.parts = [mock_part]
        return [mock_res]

    # Set side_effect of mock_run to simulate the sequence of calls:
    # 1. classifier_runner.run: {"complexity": "LOW"}
    # 2. main_runner.run: "This is a simple report on Ohio electricity rates."
    # 3. judge_runner.run: "Audit: No hallucinations found. PASSED."
    # 4. evaluator_runner.run: '{"interaction_type": "directive", "autonomy_level": 2, "human_only_time_minutes": 10, "human_education_years_required": 12, "task_success": true}'
    mock_run.side_effect = [
        make_mock_response('{"complexity": "LOW"}'),
        make_mock_response("This is a simple report on Ohio electricity rates."),
        make_mock_response("Audit: No hallucinations found. PASSED."),
        make_mock_response('{"interaction_type": "directive", "autonomy_level": 2, "human_only_time_minutes": 10, "human_education_years_required": 12, "task_success": true}')
    ]
    
    result = export_agent.query("What is the electricity rate in Ohio?")
    
    assert "This is a simple report on Ohio electricity rates." in result
    assert "Auditor Judge Verification" in result
    assert "PASSED" in result
    
    log_files = glob.glob(os.path.join(log_dir, "*.json"))
    assert len(log_files) == 1
    
    with open(log_files[0], "r") as f:
        log_data = json.load(f)
        assert log_data["query"] == "What is the electricity rate in Ohio?"
        assert log_data["primitives"]["interaction_type"] == "directive"
        assert log_data["primitives"]["autonomy_level"] == 2
        assert log_data["primitives"]["human_only_time_minutes"] == 10
        assert log_data["primitives"]["human_education_years_required"] == 12
        assert log_data["primitives"]["task_success"] is True


@patch('google.adk.runners.InMemoryRunner.run')
def test_query_routing_and_primitives_high_complexity_with_rejection(mock_run):
    from economic_research.agent import export_agent
    
    log_dir = "/Users/enriq/.gemini/jetski/scratch/observability"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        
    def make_mock_response(text):
        mock_part = MagicMock()
        mock_part.text = text
        mock_res = MagicMock()
        mock_res.content.parts = [mock_part]
        return [mock_res]

    # Sequence of calls:
    # 1. classifier_runner.run: {"complexity": "HIGH"}
    # 2. main_runner.run: "Austin is better."
    # 3. judge_runner.run: "[REJECT] Missing Raleigh comparison data."
    # 4. main_runner.run: "Austin vs Raleigh: Austin is better."
    # 5. evaluator_runner.run: '{"interaction_type": "task_iteration", "autonomy_level": 4, "human_only_time_minutes": 120, "human_education_years_required": 16, "task_success": true}'
    mock_run.side_effect = [
        make_mock_response('{"complexity": "HIGH"}'),
        make_mock_response("Austin is better."),
        make_mock_response("[REJECT] Missing Raleigh comparison data."),
        make_mock_response("Austin vs Raleigh: Austin is better."),
        make_mock_response('{"interaction_type": "task_iteration", "autonomy_level": 4, "human_only_time_minutes": 120, "human_education_years_required": 16, "task_success": true}')
    ]
    
    result = export_agent.query("Compare Austin and Raleigh for a new tech hub.")
    
    assert "Austin vs Raleigh" in result
    assert "Self-Corrected v2" in result
    assert "Missing Raleigh" in result
    
    log_files = glob.glob(os.path.join(log_dir, "*.json"))
    assert len(log_files) == 1
    
    with open(log_files[0], "r") as f:
        log_data = json.load(f)
        assert log_data["primitives"]["interaction_type"] == "task_iteration"
        assert log_data["primitives"]["autonomy_level"] == 4
        assert log_data["primitives"]["human_only_time_minutes"] == 120
        assert log_data["primitives"]["human_education_years_required"] == 16
        assert log_data["primitives"]["task_success"] is True


def test_analyze_workforce_exposure():
    from economic_research.tools.workforce_exposure_skill import analyze_workforce_exposure
    
    result = analyze_workforce_exposure(["Software Developers", "Customer Service Representatives"])
    data = json.loads(result)
    
    assert len(data) == 2
    assert data[0]["exposure_level"] == "High"
    assert "Augmentation" in data[0]["impact_mode"]
    assert "Writing/refactoring code" in data[0]["key_exposed_tasks"]
    
    assert data[1]["exposure_level"] == "High"
    assert "Automation" in data[1]["impact_mode"]

    result_unknown = analyze_workforce_exposure(["Astronaut"])
    data_unknown = json.loads(result_unknown)
    assert data_unknown[0]["exposure_level"] == "Unknown/Fuzzy Match"


def test_fetch_anthropic_economic_index_data():
    from economic_research.tools.economic_index_skill import fetch_anthropic_economic_index_data
    
    result_model = fetch_anthropic_economic_index_data("model_selection")
    data_model = json.loads(result_model)
    assert "shares_by_domain" in data_model
    assert data_model["average_overall_opus_share"] == "51%"
    
    result_tenure = fetch_anthropic_economic_index_data("user_tenure")
    data_tenure = json.loads(result_tenure)
    assert "comparison" in data_tenure
    assert data_tenure["comparison"]["personal_use_share"]["high_tenure"] == "38%"
    
    result_invalid = fetch_anthropic_economic_index_data("invalid_type")
    data_invalid = json.loads(result_invalid)
    assert "error" in data_invalid


def test_fetch_mls_property_listings():
    from economic_research.tools.mls_property_analysis_skill import fetch_mls_property_listings
    
    result = fetch_mls_property_listings("Columbus")
    data = json.loads(result)
    
    assert len(data) > 0
    assert data[0]["Property Type"] in ["Condo", "Single-family", "Multifamily"]
    assert "Estimated Cap Rate" in data[0]
    assert "Price-to-Rent Ratio" in data[0]
    
    result_invalid = fetch_mls_property_listings("London")
    data_invalid = json.loads(result_invalid)
    assert "status" in data_invalid
    assert data_invalid["status"] == "No listings found"


@patch('requests.get')
def test_fetch_hud_usps_crosswalk(mock_get):
    from economic_research.tools.hud_skill import fetch_hud_usps_crosswalk
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "results": [
                {
                    "geoid": "48453",
                    "zip": "78702",
                    "res_ratio": "0.98"
                }
            ]
        }
    }
    mock_get.return_value = mock_resp
    
    result = fetch_hud_usps_crosswalk("78702")
    data = json.loads(result)
    assert data["ZIP"] == "78702"
    assert data["County_FIPS"] == "48453"
    assert data["Residential_Ratio"] == "0.98"


@patch('requests.get')
def test_fetch_hud_chas_data(mock_get):
    from economic_research.tools.hud_skill import fetch_hud_chas_data
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {
            "geoname": "Travis County",
            "A18": "100000",
            "B3": "30000",
            "D6": "10000",
            "D9": "15000"
        }
    ]
    mock_get.return_value = mock_resp
    
    result = fetch_hud_chas_data("48453")
    data = json.loads(result)
    assert data["Geography"] == "Travis County"
    assert data["Total_Households"] == "100,000"
    assert data["Households_With_Housing_Problems_Pct"] == "30.0%"
    assert data["Households_Cost_Burdened_Pct"] == "25.0%"



def test_model_labor_shifts():
    from economic_research.tools.labor_shift_skill import model_labor_shifts
    
    result = model_labor_shifts(["Austin", "Columbus"])
    data = json.loads(result)
    
    assert len(data) == 2
    assert data[0]["City"] == "Austin"
    assert data[0]["Vulnerability Index (0-100)"] == 35
    assert data[0]["Augmentation Potential (0-100)"] == 85
    
    assert data[1]["City"] == "Columbus"
    assert data[1]["Vulnerability Index (0-100)"] == 68
    
    result_unknown = model_labor_shifts(["Orlando"])
    data_unknown = json.loads(result_unknown)
    assert data_unknown[0]["Vulnerability Index (0-100)"] == 50


def test_run_econometric_regression():
    from economic_research.tools.econometrics_skill import run_econometric_regression
    
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    x = [2.0, 4.0, 6.0, 8.0, 10.0]
    
    # Test OLS
    res_ols = run_econometric_regression(y, x, analysis_type="OLS")
    data_ols = json.loads(res_ols)
    assert data_ols["Analysis"] == "Ordinary Least Squares (OLS) Regression"
    assert data_ols["Observations"] == 5
    assert float(data_ols["R_Squared"]) == 1.0
    
    # Test Correlation
    res_corr = run_econometric_regression(y, x, analysis_type="correlation")
    data_corr = json.loads(res_corr)
    assert data_corr["Analysis"] == "Pearson Correlation Matrix"
    assert data_corr["Matrix"]["Y"]["Y"] == 1.0

    # Test ADF
    res_adf = run_econometric_regression(y, x, analysis_type="ADF")
    data_adf = json.loads(res_adf)
    assert data_adf["Analysis"] == "Augmented Dickey-Fuller (ADF) Stationarity Test"


def test_underwrite_deal_leverage():
    from economic_research.tools.underwriting_skill import underwrite_deal_leverage
    
    result = underwrite_deal_leverage(
        purchase_price=1000000.0,
        rent_monthly=10000.0,
        down_payment_pct=25.0,
        interest_rate=6.0,
        operating_expenses_pct=40.0,
        vacancy_rate=5.0
    )
    data = json.loads(result)
    assert data["Acquisition_Summary"]["Purchase_Price"] == "$1,000,000.00"
    assert data["Acquisition_Summary"]["Loan_Amount"] == "$750,000.00"
    assert data["Performance_Metrics"]["Unleveraged_Cap_Rate"] == "6.84%"
    assert data["Underwriting_Assumptions"]["Down_Payment_Pct"] == "25.0%"


def test_generate_location_scorecard():
    from economic_research.tools.scorecard_skill import generate_location_scorecard
    
    result = generate_location_scorecard(["TX", "NC"])
    data = json.loads(result)
    assert "Scorecard_Summary" in data
    assert len(data["Scorecard_Summary"]) == 2
    assert data["Scorecard_Summary"][0]["State_Code"] in ["TX", "NC"]


def test_estimate_employee_relocation():
    from economic_research.tools.relocation_skill import estimate_employee_relocation
    
    result = estimate_employee_relocation("CA", "NC", "48453", "37183", 150000.0)
    data = json.loads(result)
    assert "Relocation_Comparison" in data
    assert data["Annual_Cost_Analysis"]["Estimated_State_Income_Tax"]["Origin"] == "$19,950.00 (13.30%)"


def test_search_macro_series():
    from economic_research.tools.macro_search_skill import search_macro_series
    
    result = search_macro_series("gdp")
    data = json.loads(result)
    assert data["Query"] == "gdp"
    assert data["Matches"][0]["series_id"] == "GDPC1"


@patch('requests.get')
def test_fetch_hud_fmr_data_city_fallback(mock_get):
    from economic_research.tools.hud_skill import fetch_hud_fmr_data
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "county_name": "Travis County",
            "basicdata": {
                "fmr_2": "1800"
            }
        }
    }
    mock_get.return_value = mock_resp

    
    # Call with "Austin" city fallback instead of "48453"
    result = fetch_hud_fmr_data("Austin")
    data = json.loads(result)
    assert data["Geography"] == "Travis County"
    assert data["Rent_2BR"] == "$1,800"


@patch('requests.get')
def test_analyze_political_stability(mock_get):
    from economic_research.tools.fec_skill import analyze_political_stability
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "state": "TX",
                "cycle": 2024,
                "total": 50000000.0,
                "individual_total": 30000000.0,
                "pac_total": 20000000.0
            }
        ]
    }
    mock_get.return_value = mock_resp
    
    result = analyze_political_stability("TX")
    data = json.loads(result)
    assert data["State"] == "TX"
    assert "Total Contributions" in data








