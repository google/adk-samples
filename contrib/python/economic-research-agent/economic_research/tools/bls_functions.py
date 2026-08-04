#  Copyright 2025 Google LLC. This software is provided as-is, without warranty
#  or representation for any use or purpose. Your use of it is subject to your
#  agreement with Google.
"""Tools for Bureau of Labor Statistics (Internal Logic)."""

import json
import os
from typing import Any

import pandas as pd
from fredapi import Fred

from .tax_foundation_skill import fetch_state_tax_rates


def find_labor_force_stats(
    city_names: list[str],
) -> Any:
    """Use this tool whenever a user is looking for information on
    the labor force for a city or cities. This might include
    unemployment or labor force.

    Args:
        city_names (List[str]): A list of at least 1 city that a user
            is looking for to get labor force statistics on.

    Returns:
        labor_force_df: A Pandas Dataframe containing the labor
            stats.
    """
    # Get Labor Force & Unemployment stats.
    from economic_research.tools.fred_skill import fetch_regional_macro_stats

    macro_json = fetch_regional_macro_stats(
        city_names, series_type="unemployment"
    )

    if "ERROR" in macro_json or "No FRED data" in macro_json:
        # Fallback to empty DataFrame
        return pd.DataFrame(), {"citations": []}

    data = json.loads(macro_json)
    labor_force_df = pd.DataFrame(data)

    # Extract citations
    citations = [d["Source"] for d in data if "Source" in d]

    return labor_force_df.to_dict(orient="records"), {
        "citations": list(set(citations))
    }


def find_median_hourly_wages(
    city_names: list[str],
) -> Any:
    """Use this tool whenever a user is looking for hourly median
    wages for a city or cities.

    Args:
        city_names (List[str]): A list of at least 1 city that a user
            is looking for to get the median hourly wage for.

    Returns:
        median_hourly_wages: A Pandas Dataframe containing the hourly
            wages per hour.
    """
    fred_key = os.getenv("FRED_API_KEY")
    fred = Fred(api_key=fred_key)

    results = []
    for city in city_names:
        # Live search for occupation wages (fallback to general wages if specific failed)
        search_query = f"{city} wages"
        search_results = fred.search(search_query)
        if search_results is not None and not search_results.empty:
            series_id = search_results.iloc[0].name
            series_data = fred.get_series(series_id)
            if series_data is not None and not series_data.empty:
                val = series_data.iloc[-1]
                results.append(
                    {
                        "City": city,
                        "Median Wage": f"${val:.2f}",
                        "Source": f"FRED ({series_id})",
                    }
                )

    if not results:
        return [{"Message": "No wage data found via FRED live search."}], {
            "citations": []
        }

    return results, {"citations": [r["Source"] for r in results]}


def find_state_union_employment(
    state_names: list[str],
) -> Any:
    """Use this tool whenever a user is looking for union
    employment rates for a state.

    Args:
        state_names (List[str]): A list of at least 1 state that a user
            is looking for to get the union employment rate for.

    Returns:
        union_employment_rate: A Pandas Dataframe containing the hourly
            state union employment rates.
    """
    fred_key = os.getenv("FRED_API_KEY")
    fred = Fred(api_key=fred_key)

    results = []
    for state in state_names:
        search_query = f"{state} union membership percentage"
        search_results = fred.search(search_query)
        if search_results is not None and not search_results.empty:
            series_id = search_results.iloc[0].name
            series_data = fred.get_series(series_id)
            if series_data is not None and not series_data.empty:
                val = series_data.iloc[-1]
                results.append(
                    {
                        "State": state,
                        "Union Membership %": f"{val:.1f}%",
                        "Source": f"FRED ({series_id})",
                    }
                )

    if not results:
        return [{"Message": "No union data found via FRED live search."}], {
            "citations": []
        }

    return results, {"citations": [r["Source"] for r in results]}


def find_state_tax_rate(
    state_names: list[str],
) -> Any:
    """Use this tool whenever a user is looking for tax
    rates for a state.

    Args:
        state_names (List[str]): A list of at least 1 state that a user
            is looking for to get the tax rate for a state.

    Returns:
        state_tax_rate_df: A Pandas Dataframe containing the hourly
            state union employment rates.
    """
    # Directly fetch from Tax Foundation Scraper (Live-API Strategy)
    tax_json = fetch_state_tax_rates(state_names)
    data = json.loads(tax_json)

    # Extract citations
    citations = [d["Source"] for d in data if "Source" in d]

    return data, {"citations": list(set(citations))}
