#  Copyright 2025 Google LLC. This software is provided as-is, without warranty
#  or representation for any use or purpose. Your use of it is subject to your
#  agreement with Google.
"""ADK Skill: FRED Macro Data (St. Louis Fed). Replaces BigQuery with direct API calls."""

import json
import os

from fredapi import Fred
from pydantic import BaseModel, Field


class FredRegionalRequest(BaseModel):
    city_names: list[str] = Field(
        ...,
        description="List of city names to fetch unemployment/macro data for.",
    )
    series_type: str = Field(
        "unemployment",
        description="Type of data to fetch: unemployment, gdp, or residential_construction.",
    )


def fetch_regional_macro_stats(
    city_names: list[str], series_type: str = "unemployment"
) -> str:
    """
    Fetches regional economic metrics directly from St. Louis Fed (FRED) API.
    Replaces legacy BigQuery labor tables. Support MSAs like Austin, Raleigh, etc.
    """
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        return "ERROR: FRED_API_KEY is not set in environment variables."

    fred = Fred(api_key=fred_key)

    # Simple mapping logic for top MSAs (Can be expanded with dynamic search)
    # Series IDs follow a pattern: [MSA CODE]UR for unemployment.
    msa_codes = {
        "Austin": "AUST448",  # Austin-Round Rock MSA
        "Raleigh": "RALE937",  # Raleigh-Cary MSA
        "San Francisco": "SANF806",
        "Dallas": "DALL148",
        "Denver": "DENN508",
        "Seattle": "SEAT653",
        "Atlanta": "ATLA013",
        "Charlotte": "CHAL837",
    }

    series_suffixes = {
        "unemployment": "UR",
        "gdp": "RGDP",  # Real GDP
        "residential_construction": "BP1FH",  # Building Permits 1-Unit
    }

    results = []

    for city in city_names:
        city_clean = city.split(",")[0].strip()  # Handle "Austin, TX"
        code = msa_codes.get(city_clean)

        if not code:
            # Plan B: Search for the MSA series
            search_query = f"{city_clean} unemployment rate"
            search_results = fred.search(search_query)
            if not search_results.empty:
                code = search_results.iloc[0].name  # Use the most relevant ID
            else:
                continue

        # Series construction logic
        suffix = series_suffixes.get(series_type, "UR")
        series_id = f"{code}{suffix}"

        try:
            try:
                data_series = fred.get_series(series_id)
            except Exception:
                # If hard-coded ID fails, use search as fallback
                search_map = {
                    "residential_construction": "building permits",
                    "unemployment": "unemployment rate",
                    "gdp": "real gdp",
                }
                query_topic = search_map.get(series_type, series_type)
                search_query = f"{city_clean} {query_topic}"
                search_results = fred.search(search_query)
                if not search_results.empty:
                    series_id = search_results.iloc[0].name
                    data_series = fred.get_series(series_id)
                else:
                    continue

            if not data_series.empty:
                latest_val = data_series.iloc[-1]
                latest_date = data_series.index[-1].strftime("%Y-%m-%d")

                # Sample 10 annual data points (step by 12 for monthly data, or 1 for annual)
                historical_data = []
                step = (
                    12 if len(data_series) > 24 else 1
                )  # Simple heuristic: if monthly (len > 24), step by 12.
                subset = data_series.iloc[
                    -120::step
                ]  # Take last 120 points (e.g. 10 years of monthly data)

                for idx, val in subset.items():
                    historical_data.append(
                        {
                            "date": idx.strftime("%Y-%m-%d"),
                            "value": f"{val:.2f}%"
                            if "unemployment" in series_type
                            else f"{val:,.2f}",
                        }
                    )

                results.append(
                    {
                        "City": city_clean,
                        "Metric": series_type.capitalize(),
                        "Latest Value": f"{latest_val:.2f}%"
                        if "unemployment" in series_type
                        else f"{latest_val:,.2f}",
                        "Latest Date": latest_date,
                        "Historical_10_Year_Points": historical_data,
                        "Source": f"FRED ({series_id})",
                    }
                )
        except Exception:
            continue

    if not results:
        return f"No FRED data found for the requested cities: {city_names}."

    # Return as JSON string for Scribe node processing
    return json.dumps(results, indent=2)
