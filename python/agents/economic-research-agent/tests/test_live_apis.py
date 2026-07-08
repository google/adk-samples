# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""Live API Testing Script to verify real configured credentials and endpoints."""

import sys
from dotenv import load_dotenv

load_dotenv()

from economic_research.tools.hud_skill import (
    fetch_hud_fmr_data,
    fetch_hud_income_limits,
    fetch_hud_usps_crosswalk,
    fetch_hud_chas_data
)
from economic_research.tools.bea_skill import fetch_bea_regional_data
from economic_research.tools.eia_skill import fetch_state_electricity_rates
from economic_research.tools.fred_skill import fetch_regional_macro_stats
from economic_research.tools.tax_foundation_skill import fetch_state_tax_rates
from economic_research.tools.fec_skill import analyze_political_stability
from economic_research.tools.bls_api_skill import analyze_labor_force_quality


def main():
    print("====================================================")
    print("               LIVE API INTEGRATION TESTS           ")
    print("====================================================")
    
    # 1. HUD FMR
    print("\n[1] Testing HUD FMR (Travis County, TX)...")
    res_fmr = fetch_hud_fmr_data("48453")
    print(res_fmr)
    
    # 2. HUD Income Limits
    print("\n[2] Testing HUD Income Limits (Travis County, TX)...")
    res_il = fetch_hud_income_limits("48453")
    print(res_il)
    
    # 3. USPS ZIP Crosswalk
    print("\n[3] Testing USPS Crosswalk (78702 ZIP -> County FIPS)...")
    res_cw = fetch_hud_usps_crosswalk("78702")
    print(res_cw)
    
    # 4. HUD CHAS Data
    print("\n[4] Testing HUD CHAS Data (Travis County, TX)...")
    res_chas = fetch_hud_chas_data("48453")
    print(res_chas)
    
    # 5. BEA GDP
    print("\n[5] Testing BEA GDP (Austin)...")
    res_bea = fetch_bea_regional_data(["Austin"], "GDP")
    print(res_bea)
    
    # 6. EIA Electricity Rates
    print("\n[6] Testing EIA Electricity Rates (TX)...")
    res_eia = fetch_state_electricity_rates(["TX"])
    print(res_eia)
    
    # 7. FRED Macro Stats
    print("\n[7] Testing FRED Macro Stats (Unemployment)...")
    res_fred = fetch_regional_macro_stats(["unemployment_rate"])
    print(res_fred)
    
    # 8. Tax Foundation
    print("\n[8] Testing Tax Foundation (NC Tax Brackets)...")
    res_tax = fetch_state_tax_rates(["North Carolina"])
    print(res_tax)

    
    # 9. FEC Campaign Contributions
    print("\n[9] Testing FEC Campaign Contributions (TX)...")
    res_fec = analyze_political_stability("TX")
    print(res_fec)
    
    # 10. BLS Labor Force
    print("\n[10] Testing BLS Labor Force (NC)...")
    res_bls = analyze_labor_force_quality("NC")
    print(res_bls)

    print("\n====================================================")
    print("                    TESTS COMPLETE                  ")
    print("====================================================")


if __name__ == "__main__":
    main()
