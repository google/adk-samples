#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Site Selection & Commercial Real Estate (CoStar/Zillow/Redfin)."""

import json

from pydantic import BaseModel, Field


class RealEstateRequest(BaseModel):
    city_names: list[str] = Field(
        ...,
        description="List of city names to fetch real estate benchmarks for.",
    )
    property_type: str = Field(
        "Office",
        description="Type of property: Office, Industrial, or Logistics.",
    )


def get_real_estate_roi(
    city_names: list[str], property_type: str = "Office"
) -> str:
    """
    Fetches commercial lease rates and availability from CoStar/Zillow/Redfin data benchmarks.
    Site selection depends on the P&L of the building, not just the labor.
    """
    # 1. Fetch MSA-level property benchmarks
    # Note: These are usually retrieved from a 'Real Estate' BigQuery table or a direct CoStar API.
    # Current implementation provides grounded benchmarks for site-selection comparison.
    results = []

    for city in city_names:
        city_clean = city.split(",")[0].strip()

        from economic_research.tools.dynamic_search_harvester import harvest_real_estate_roi
        harvested = harvest_real_estate_roi(city_clean, property_type)
        results.append(harvested)

    return json.dumps(results, indent=2)
