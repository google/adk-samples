#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Policy & Political Risk (OpenSecrets). Governance ROI."""

import json

from pydantic import BaseModel, Field


class PolicyRequest(BaseModel):
    state_names: list[str] = Field(
        ...,
        description="List of states to fetch political/policy risk benchmarks for.",
    )


def get_policy_risk_benchmarks(state_names: list[str]) -> str:
    """
    Fetches OpenSecrets benchmarks for regional political stability and corporate lobbying climates.
    Identifies 10-year regulatory risk for capital-intensive site selections.
    """
    results = []

    for state in state_names:
        # Grounded OpenSecrets Benchmarks (Sample data mapping)
        policy_data = {
            "Texas": {
                "lobby_climate": "Very Pro-Business",
                "regulatory_stability": "High",
                "upcoming_tax_sunsets": "Inventory Tax (2026)",
            },
            "North Carolina": {
                "lobby_climate": "Pro-Business",
                "regulatory_stability": "Very High",
                "upcoming_tax_sunsets": "None",
            },
            "California": {
                "lobby_climate": "Mixed/Regulated",
                "regulatory_stability": "Moderate",
                "upcoming_tax_sunsets": "R&D Credit (Expected)",
            },
        }

        data = policy_data.get(
            state, {"lobby_climate": "N/A", "regulatory_stability": "N/A"}
        )

        results.append(
            {
                "State": state,
                "Lobbying Climate Cluster": data["lobby_climate"],
                "Regulatory Stability Index": data["regulatory_stability"],
                "Upcoming Policy Sunsets": data.get(
                    "upcoming_tax_sunsets", "N/A"
                ),
                "Source": "OpenSecrets & State Chamber Governance Grounding",
            }
        )

    return json.dumps(results, indent=2)


#  Copyright 2025 Google LLC.
"""ADK Skill: Cost of Living & Purchasing Power (C2ER). Talent ROI."""


class ColRequest(BaseModel):
    city_names: list[str] = Field(
        ...,
        description="List of city names to fetch cost-of-living adjustments (COLA) for.",
    )


def get_purchasing_power_adjustment(city_names: list[str]) -> str:
    """
    Fetches C2ER benchmarks for Cost of Living Index (COLI) and effective purchasing power.
    Analyzes whether a 'higher wage' in one city results in more 'real income' after housing and utilities.
    """
    results = []

    for city in city_names:
        city_clean = city.split(",")[0].strip()

        # Grounded C2ER COLI Benchmarks (Sample data mapping)
        coli_data = {
            "Austin": {
                "coli_index": "104.2",
                "housing_premium": "+15%",
                "real_income_multiplier": "0.96",
            },
            "Raleigh": {
                "coli_index": "92.4",
                "housing_premium": "-5%",
                "real_income_multiplier": "1.08",
            },
            "San Francisco": {
                "coli_index": "178.5",
                "housing_premium": "+110%",
                "real_income_multiplier": "0.56",
            },
        }

        data = coli_data.get(
            city_clean, {"coli_index": "N/A", "real_income_multiplier": "1.0"}
        )

        results.append(
            {
                "City": city_clean,
                "COLI Index (National=100)": data["coli_index"],
                "Real Income Multiplier": data["real_income_multiplier"],
                "Note": "A multiplier > 1.0 means your dollar goes further here than the national average.",
                "Source": "C2ER (Council for Community & Economic Research) Grounding",
            }
        )

    return json.dumps(results, indent=2)
