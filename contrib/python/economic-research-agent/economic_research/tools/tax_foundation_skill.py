#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Tax Foundation Scraper. Real-time state corporate tax data."""

import json

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class TaxRequest(BaseModel):
    state_names: list[str] = Field(
        ...,
        description="List of full state names (e.g., ['Texas', 'California']).",
    )


def fetch_state_tax_rates(state_names: list[str]) -> str:
    """
    Scrapes Tax Foundation for the latest state corporate income tax rates.
    This bypasses legacy BigQuery dependencies and provides real-time data.
    """
    url = "https://taxfoundation.org/data/all/state/state-corporate-income-tax-rates-brackets-2024/"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # The Tax Foundation often uses a table with class 'table' or inside a specific div
        # Let's find all tables and look for one containing state names
        tables = soup.find_all("table")
        tax_data = {}

        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all(["td", "th"])
                if len(cols) >= 2:
                    state_raw = cols[0].get_text(strip=True)
                    rate_raw = cols[1].get_text(strip=True)

                    # Clean up footnote references like 'Alaska (a)'
                    state_clean = state_raw.split("(")[0].strip()
                    tax_data[state_clean] = rate_raw

        results = []
        for state in state_names:
            rate = tax_data.get(state, "N/A (Check Source)")
            results.append(
                {
                    "State": state,
                    "Corporate Tax Rate": rate,
                    "Source": "Tax Foundation (Live Scrape 2024)",
                }
            )

        return json.dumps(results, indent=2)

    except Exception as e:
        # Fallback to a known list if scraping fails (Hardening)
        fallback_rates = {
            "Texas": "None (Gross Receipts Tax)",
            "California": "8.84%",
            "New York": "7.25%",
            "Florida": "5.5%",
            "Illinois": "9.5%",
            "Pennsylvania": "8.49%",
            "Ohio": "None (Gross Receipts Tax)",
            "Washington": "None (Gross Receipts Tax)",
            "North Carolina": "2.5%",
            "Arizona": "4.9%",
        }

        results = []
        for state in state_names:
            results.append(
                {
                    "State": state,
                    "Corporate Tax Rate": fallback_rates.get(state, "N/A"),
                    "Source": "Tax Foundation (Fallback/Hardcoded)",
                    "Error": str(e)
                    if "404" not in str(e)
                    else "Page structure changed",
                }
            )
        return json.dumps(results, indent=2)


if __name__ == "__main__":
    # Test
    print(fetch_state_tax_rates(["Texas", "California", "Minnesota"]))
