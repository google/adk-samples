#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Political Stability & Campaign Finance (FEC API)."""

import json
import os

import requests
from pydantic import BaseModel, Field

class FECRequest(BaseModel):
    state_abbr: str = Field(
        ..., description="Two-letter state abbreviation (e.g., 'TX')."
    )
    cycle: str = Field("2024", description="Election cycle year to analyze.")


def analyze_political_stability(state_abbr: str, cycle: str = "2024") -> str:
    """
    Fetches Campaign Finance (FEC) contribution data for a specific state.
    Provides site selection agents with a metric for political stability and business alignment.
    High PAC activity often correlates with high regulatory engagement or a shifting political climate.
    """
    key = os.getenv("FEC_API_KEY", "").strip() or "DEMO_KEY"
    
    # FEC Endpoint: Contributions by State and Cycle
    url = "https://api.open.fec.gov/v1/totals/by_state/"
    params = {
        "api_key": key,
        "state": state_abbr,
        "cycle": cycle,
        "per_page": 1,
    }

    try:
        response = requests.get(url, params=params, timeout=12)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])

            if not results:
                return json.dumps(
                    {"ERROR": f"No FEC data found for state: {state_abbr}"},
                    indent=2,
                )

            entry = results[0]

            summary = {
                "State": state_abbr,
                "Election Cycle": cycle,
                "Total Contributions": f"${entry.get('receipts', 0):,.2f}",
                "Political Activity Level": "High"
                if entry.get("receipts", 0) > 50000000
                else "Moderate",
                "Source": "U.S. Federal Election Commission (FEC) API",
            }
            return json.dumps(summary, indent=2)
        else:
            return json.dumps(
                {"ERROR": f"FEC API status {response.status_code}"}, indent=2
            )

    except Exception as e:
        return json.dumps({"ERROR": str(e)}, indent=2)
