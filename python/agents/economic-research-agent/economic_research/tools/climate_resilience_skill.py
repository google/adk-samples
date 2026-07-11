#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Climate Risk & Resilience (FEMA NRI). 20-year investment protection."""

import json

from pydantic import BaseModel, Field


class ClimateRequest(BaseModel):
    city_names: list[str] = Field(
        ..., description="List of cities to fetch climate risk benchmarks for."
    )


def get_climate_risk_index(city_names: list[str]) -> str:
    """
    Fetches FEMA National Risk Index (NRI) benchmarks for MSAs.
    Analyzes 18 natural hazards (Heat, Flood, Hurricane) to protect 20-year infrastructure investments.
    """
    results = []

    for city in city_names:
        city_clean = city.split(",")[0].strip()

        from economic_research.tools.dynamic_search_harvester import harvest_climate_risk
        harvested = harvest_climate_risk(city)
        results.append(harvested)

    return json.dumps(results, indent=2)
