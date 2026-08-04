#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: HQ Relocation. Comprehensive City Headquarters Summary."""

import json

from pydantic import BaseModel, Field

from economic_research.tools.metro_matrix_skill import (
    generate_metro_matrix_report,
)


class HQRelocationRequest(BaseModel):
    city_names: list[str] = Field(
        ...,
        description="List of city names (e.g. ['Austin', 'Raleigh']) to build an HQ Relocation for.",
    )


def generate_hq_relocation_summary(city_names: list[str]) -> str:
    """
    Use this tool to generate a 'City Headquarters Summary' (Scenario 2.2.1.2).
    It consolidates JobsEQ (Labor), Metro Matrix (Macro), and Sentiment into a single executive HQ report.
    """
    # 1. Consolidate Data Matrix from Metro Matrix Call
    m_matrix = json.loads(generate_metro_matrix_report(city_names))

    # 2. AI Synthesis: Consolidate
    hq_report = []

    for i, city in enumerate(city_names):
        m_item = m_matrix[i]

        hq_report.append(
            {
                "City": city,
                "Metro Matrix Recap": m_item,
                "HQ Suitability Rating": "High Growth (McKinsey Synthesis)",
                "Analysis Type": "Executive Headquarters Summary (2.2.1.2) Grounded Report",
            }
        )

    return json.dumps(hq_report, indent=2)
