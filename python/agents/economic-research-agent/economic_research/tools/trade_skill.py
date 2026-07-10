#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: USITC Trade Data. Regional Import/Export dependencies."""

import os
import json
import requests

from pydantic import BaseModel, Field


class TradeRequest(BaseModel):
    state_names: list[str] = Field(
        ..., description="List of states to fetch trade dependency data for."
    )
    commodity: str = Field(
        "Electronic Products",
        description="HS Code or Commodity name (e.g. 'Semiconductors', 'Auto parts').",
    )


HS_CODE_MAP = {
    "Electronic Products": "85",
    "Semiconductors": "85",
    "Electrical Machinery": "85",
    "Industrial Machinery": "84",
    "Machinery": "84",
    "Pharmaceuticals": "30",
    "Agricultural Products": "12",
}

STATE_MAP = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

def fetch_regional_trade_data(
    state_names: list[str], commodity: str = "Electronic Products"
) -> str:
    """
    Fetches international trade flow data for specific states and commodities.
    Essential for analyzing supply-chain resilience and industry clustering.
    """
    results = []
    census_key = os.getenv("CENSUS_API_KEY", "").strip()
    
    # Normalize commodity name to handle case variations (e.g. "pharmaceuticals" -> "Pharmaceuticals")
    comm_clean = commodity.strip().title()
    
    # 1. Fallback Offline Data Bank
    trade_bank = {
        "Texas": {
            "Electronic Products": "Top Import (Mexico), $45B annual value",
            "Industrial Machinery": "$30B annual export",
        },
        "California": {
            "Electronic Products": "Global Hub, $60B annual flux",
            "Agricultural Products": "$15B annual export",
        },
        "North Carolina": {
            "Pharmaceuticals": "Major Manufacturing Hub, $8B annual export"
        },
        "Arizona": {
            "Semiconductors": "$12B annual state-origin export",
            "Electronic Products": "$12B annual state-origin export (Semiconductors)"
        },
    }

    # 2. Live API Sourcing
    if census_key:
        hs_code = HS_CODE_MAP.get(comm_clean)
        if hs_code:
            url = "https://api.census.gov/data/timeseries/intltrade/exports/statehs"
            params = {
                "get": "STATE,ALL_VAL_YR,E_COMMODITY",
                "E_COMMODITY": hs_code,
                "time": "2024",
                "key": census_key
            }
            try:
                r = requests.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    rows = data[1:]
                    
                    for state in state_names:
                        state_abbr = STATE_MAP.get(state)
                        if not state_abbr:
                            continue
                            
                        matched_row = None
                        for row in rows:
                            if row[0] == state_abbr:
                                matched_row = row
                                break
                                
                        if matched_row:
                            value_usd = int(matched_row[1])
                            time_period = matched_row[4]
                            
                            if value_usd >= 1_000_000_000:
                                val_str = f"${value_usd / 1_000_000_000:.2f}B"
                            else:
                                val_str = f"${value_usd / 1_000_000:.2f}M"
                                
                            results.append({
                                "State": state,
                                "Commodity": comm_clean,
                                "Market Profile": f"YTD Export Value: {val_str} (cumulative through {time_period})",
                                "Source": "U.S. Census Bureau International Trade API (statehs)"
                            })
                            continue
            except Exception as e:
                print(f"⚠️ Census trade API call failed: {e}. Falling back to sandbox database.")

    # 3. Apply offline fallback for any states that failed or weren't resolved live
    for state in state_names:
        if any(res.get("State") == state for res in results):
            continue
            
        data = trade_bank.get(state, {}).get(
            comm_clean,
            "Data unavailable in trade database.",
        )
        results.append(
            {
                "State": state,
                "Commodity": comm_clean,
                "Market Profile": data,
                "Source": "USITC DataWeb (Regional Trade Flows) - Sandbox",
            }
        )

    return json.dumps(results, indent=2)

