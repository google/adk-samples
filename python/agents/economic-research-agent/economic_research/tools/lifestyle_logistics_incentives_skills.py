#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Logistics & Transit Efficiency (DOT/BTS). Supply Chain Grounding."""

import json

from pydantic import BaseModel, Field


class LogisticsRequest(BaseModel):
    city_names: list[str] = Field(
        ...,
        description="List of city names to analyze logistics and shipping costs for.",
    )


def get_logistics_efficiency(city_names: list[str]) -> str:
    """
    Fetches DOT (Bureau of Transportation Stats) benchmarks for MSA-to-MSA shipping costs and transit times.
    Essential for supply chain optimization in manufacturing relocations.
    """
    results = []

    for city in city_names:
        city_clean = city.split(",")[0].strip()

        from economic_research.tools.dynamic_search_harvester import harvest_logistics_efficiency
        harvested = harvest_logistics_efficiency(city)
        results.append(harvested)

    return json.dumps(results, indent=2)


#  Copyright 2025 Google LLC.
"""ADK Skill: Lifestyle Density & Amenity Scoring (Google Places/WalkScore). Talent Retention."""


class LifestyleRequest(BaseModel):
    city_names: list[str] = Field(
        ..., description="List of city names to fetch lifestyle benchmarks for."
    )


def get_cultural_amenity_score(city_names: list[str]) -> str:
    """
    Fetches Google Places and WalkScore benchmarks for 'Lifestyle ROI'.
    Talent retention depends on proximity to coffee shops, gyms, parks, and schools.
    """
    results = []

    for city in city_names:
        city_clean = city.split(",")[0].strip()

        from economic_research.tools.dynamic_search_harvester import harvest_cultural_amenities
        harvested = harvest_cultural_amenities(city)
        results.append(harvested)

    return json.dumps(results, indent=2)


#  Copyright 2025 Google LLC.
"""ADK Skill: Economic Incentives & Subsidy Discovery (Good Jobs First)."""


class IncentiveRequest(BaseModel):
    state_names: list[str] = Field(
        ...,
        description="List of states to fetch tax incentive/subsidy benchmarks for.",
    )


def get_regional_tax_incentives(state_names: list[str]) -> str:
    """
    Fetches state-level economic development incentives and active subsidy programs.
    Proactively discovers tax breaks (e.g., Chapter 313) to boost relocation ROI.
    """
    results = []

    for state in state_names:
        from economic_research.tools.dynamic_search_harvester import harvest_regional_incentives
        harvested = harvest_regional_incentives(state)
        results.append(harvested)

    return json.dumps(results, indent=2)
