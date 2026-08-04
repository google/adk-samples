#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Metro Matrix (2.2.1.1). Comprehensive MSA-level benchmarking."""

import json

from economic_research.tools.fred_skill import fetch_regional_macro_stats
from economic_research.tools.macro_foundation_skill import (
    get_state_macro_health,
)
from economic_research.tools.sentiment_skill import analyze_market_sentiment


def get_state_from_city(city: str) -> str:
    """Heuristic to map city names to their primary states."""
    lower_city = city.lower()
    if "," in city:
        return city.rsplit(",", maxsplit=1)[-1].strip()

    # Common Site Selection Hubs
    mapping = {
        "austin": "Texas",
        "dallas": "Texas",
        "houston": "Texas",
        "raleigh": "North Carolina",
        "durham": "North Carolina",
        "charlotte": "North Carolina",
        "nashville": "Tennessee",
        "memphis": "Tennessee",
        "denver": "Colorado",
        "boulder": "Colorado",
        "seattle": "Washington",
        "san francisco": "California",
        "atlanta": "Georgia",
        "phoenix": "Arizona",
    }

    for k, v in mapping.items():
        if k in lower_city:
            return v
    return "Texas"  # Default fallback for site selection demo


def generate_metro_matrix_report(city_names: list[str]) -> str:
    """
    Use this tool to generate a comprehensive 'Metro Matrix Report' (Scenario 2.2.1.1).
    """
    # 1. Gather Macro Health (BEA/Census)
    # Correct state derivation is critical for Live-API grounding accuracy.
    states = list({get_state_from_city(city) for city in city_names})
    macro_json = get_state_macro_health(states)
    macro_data = (
        json.loads(macro_json) if not macro_json.startswith("ERROR") else []
    )

    # 2. Gather Labor Stats (Live FRED)
    labor_json = fetch_regional_macro_stats(
        city_names, series_type="unemployment"
    )
    labor_data = (
        json.loads(labor_json)
        if not labor_json.startswith("No FRED data")
        and not labor_json.startswith("ERROR")
        else []
    )

    # 3. Gather Business Climate Sentiment (Search/NewsAPI)
    sentiment_data = []
    for city in city_names:
        sentiment_data.append(
            {
                "City": city,
                "Business Climate News": analyze_market_sentiment(
                    f"{city} business climate Forbes Forbes 500"
                ),
            }
        )

    # 4. AI Synthesis: Consolidate into Matrix Structure
    matrix = []
    for i, city in enumerate(city_names):
        city_clean = city.split(",")[0].strip()
        state_target = get_state_from_city(city)

        # High-fidelity target matching
        m_item = next(
            (
                m
                for m in macro_data
                if m["State"].lower() == state_target.lower()
            ),
            macro_data[0] if macro_data else {"Message": "No Macro Data"},
        )
        l_item = next(
            (
                labor
                for labor in labor_data
                if labor.get("City", "").lower() == city_clean.lower()
                or labor.get("City", "").lower() in city.lower()
            ),
            {"City": city_clean, "Message": "No Labor Data"},
        )
        s_item = sentiment_data[i]

        matrix.append(
            {
                "City": city,
                "Macro Context": m_item,
                "Labor Context": l_item,
                "Sentiment Summary": s_item["Business Climate News"],
                "Analysis Type": "Metro Matrix (2.2.1.1) Grounded Report",
            }
        )

    return json.dumps(matrix, indent=2)
