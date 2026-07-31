#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ERA MCP Server. Exposes ADK Economic Research tools to any MCP client."""

from mcp.server.fastmcp import FastMCP

# Import specialized skills
from economic_research.tools.bls_api_skill import (
    fetch_bls_series_data,
)
from economic_research.tools.census_skill import fetch_census_education_stats
from economic_research.tools.fec_skill import analyze_political_stability
from economic_research.tools.fred_skill import fetch_regional_macro_stats
from economic_research.tools.hud_skill import analyze_housing_affordability
from economic_research.tools.political_climate_skill import (
    search_lobbying_influence,
)
from economic_research.tools.regulatory_skill import fetch_regulatory_notices
from economic_research.tools.tax_foundation_skill import fetch_state_tax_rates
from economic_research.tools.trade_skill import fetch_regional_trade_data

# Initialize FastMCP Server
mcp = FastMCP("EconomicResearchAgent")


# Register Tools via MCP Decorators
@mcp.tool()
def get_macro_stats(cities: list[str], series_type: str = "unemployment"):
    """Fetches regional macro data (unemployment, GDP, construction) from FRED."""
    return fetch_regional_macro_stats(cities, series_type)


@mcp.tool()
def get_education_stats(state_abbr: str, county_code: str | None = None):
    """Fetches ACS educational attainment data from Census."""
    return fetch_census_education_stats(state_abbr, county_code)


@mcp.tool()
def analyze_affordability(county_code: str):
    """Correlates Fair Market Rent vs. Area Median Income (McKinsey-style analysis)."""
    return analyze_housing_affordability(county_code)


@mcp.tool()
def get_tax_rates(states: list[str]):
    """Scrapes latest state corporate income tax rates from Tax Foundation."""
    return fetch_state_tax_rates(states)


@mcp.tool()
def get_trade_dependency(
    states: list[str], commodity: str = "Electronic Products"
):
    """Fetches regional trade and supply-chain dependency data from USITC/EIA."""
    return fetch_regional_trade_data(states, commodity)


@mcp.tool()
def check_regulatory_notices(states: list[str], topic: str = "Semiconductor"):
    """Tracks live regulatory notices and upcoming policy shifts from Federal Register."""
    return fetch_regulatory_notices(states, topic)


@mcp.tool()
def analyze_lobbying_influence(industry: str, state: str):
    """Benchmarks political/lobbying influence from U.S. Senate LDA database."""
    return search_lobbying_influence(industry, state)


@mcp.tool()
def get_political_stability(state_abbr: str, cycle: str = "2024"):
    """Fetches Campaign Finance (FEC) totals to analyze regional political stability."""
    return analyze_political_stability(state_abbr, cycle)


@mcp.tool()
def get_labor_series(
    series_ids: list[str], start_year: str = "2023", end_year: str = "2024"
):
    """Fetches live labor statistics (Unemployment, Wages) directly from BLS API."""
    return fetch_bls_series_data(series_ids, start_year, end_year)


if __name__ == "__main__":
    mcp.run()
