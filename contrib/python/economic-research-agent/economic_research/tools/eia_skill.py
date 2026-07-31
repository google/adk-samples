#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: EIA Energy Data (U.S. Energy Information Administration)."""

import json
import logging
import os

import requests

# Configure basic logging
logger = logging.getLogger(__name__)

def fetch_state_electricity_rates(
    state_codes: list[str], sector: str = "industrial"
) -> str:
    """
    Fetches real-time average electricity prices per kWh from the EIA Open Data API.
    Crucial for calculating the operational ROI of data centers or manufacturing plants.
    """
    h_eia_key = os.getenv("EIA_API_KEY", "").strip()
    eia_key = h_eia_key.replace('"', "").replace("'", "")
    if not eia_key:
        return json.dumps(
            {"ERROR": "EIA_API_KEY not found in environment."}, indent=2
        )

    results = []
    # Map sector name to EIA v2 sectorid
    sector_map = {
        "industrial": "IND",
        "commercial": "COM",
        "residential": "RES",
    }
    s_id = sector_map.get(sector.lower(), "industrial")

    for state in state_codes:
        state = state.upper().strip()
        url = (
            f"https://api.eia.gov/v2/electricity/retail-sales/data/?api_key={eia_key}"
            f"&frequency=monthly&data[0]=price"
            f"&facets[stateid][]={state}"
            f"&facets[sectorid][]={s_id}"
            f"&sort[0][column]=period&sort[0][direction]=desc&length=1"
        )

        try:
            response = requests.get(url, timeout=12)
            if response.status_code == 200:
                full_data = response.json()
                # EIA v2 often wraps data in 'response' -> 'data'
                data_list = full_data.get("response", {}).get("data", [])
                if not data_list:
                    # Fallback for alternative v2 structures or 'ALL' sectors
                    data_list = full_data.get("data", [])

                if data_list:
                    latest = data_list[0]
                    results.append(
                        {
                            "State": state,
                            "Sector": sector.capitalize(),
                            "Avg Price (cents/kWh)": f"{float(latest.get('price', 0)):.2f}",
                            "Period": latest.get("period", "Unknown"),
                            "Source": "U.S. Energy Information Administration (EIA v2)",
                        }
                    )
                else:
                    results.append(
                        {
                            "State": state,
                            "Status": "No specific sector data found.",
                        }
                    )
            else:
                results.append(
                    {
                        "State": state,
                        "Status": f"EIA API failure ({response.status_code})",
                    }
                )
        except Exception as e:
            results.append({"State": state, "Status": f"Error: {e!s}"})

    if not results:
        return json.dumps(
            {"ERROR": f"No EIA data retrieved for {state_codes}"}, indent=2
        )

    return json.dumps(results, indent=2)
