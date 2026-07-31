#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Federal Register (Regulatory Risk tracking). Monitoring notices and rules."""

import json

import requests
from pydantic import BaseModel, Field


class RegulatoryRequest(BaseModel):
    state_names: list[str] = Field(
        ..., description="List of states to check for regulatory activities."
    )
    industry_topic: str = Field(
        "Semiconductor",
        description="Industry or topic to track (e.g. Energy, Zoning, Healthcare).",
    )


def fetch_regulatory_notices(
    state_names: list[str], industry_topic: str = "Semiconductor"
) -> str:
    """
    Fetches live regulatory filings from the Federal Register API.
    Essential for identifying legal risks and upcoming state policy shifts.
    """
    results = []

    try:
        for state in state_names:
            # Query Federal Register for the state + industry/topic
            # Example API: https://www.federalregister.gov/api/v1/documents.json
            query = f"{state} {industry_topic}"
            url = f"https://www.federalregister.gov/api/v1/documents.json?conditions[term]={query}&per_page=5"

            response = requests.get(url, timeout=12)
            if response.status_code == 200:
                data = response.json()
                filings = data.get("results", [])

                state_results = []
                for f in filings:
                    state_results.append(
                        {
                            "Title": f.get("title"),
                            "Action": f.get("action"),
                            "Date": f.get("publication_date"),
                            "URL": f.get("html_url"),
                            "Agency": f.get("agency_names", ["N/A"])[0],
                        }
                    )

                results.append(
                    {
                        "State": state,
                        "Industry/Topic": industry_topic,
                        "Notices": state_results
                        if state_results
                        else "No recent filings found.",
                        "Source": "Federal Register (Live API)",
                    }
                )
            else:
                results.append(
                    {"State": state, "ERROR": f"Status {response.status_code}"}
                )

        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"ERROR": str(e)}, indent=2)
