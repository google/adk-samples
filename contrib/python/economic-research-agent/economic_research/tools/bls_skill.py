#  Copyright 2025 Google LLC. This software is provided as-is, without warranty
#  or representation for any use or purpose. Your use of it is subject to your
#  agreement with Google.
"""ADK Skill: Bureau of Labor Statistics (BLS). Hardened labor analytics."""

from typing import Any

from pydantic import BaseModel, Field

from .bls_functions import (
    find_labor_force_stats,
    find_median_hourly_wages,
    find_state_tax_rate,
    find_state_union_employment,
)


class CityNamesRequest(BaseModel):
    city_names: list[str] = Field(
        ..., min_length=1, description="List of city names."
    )


class StateNamesRequest(BaseModel):
    state_names: list[str] = Field(
        ..., min_length=1, description="List of full state names."
    )


def labor_force_stats_skill(city_names: list[str]) -> Any:
    """
    Fetches BLS data for labor force statistics (unemployment, labor force).
    """
    return find_labor_force_stats(city_names)


def median_hourly_wages_skill(city_names: list[str]) -> Any:
    """
    Fetches BLS data for median hourly wages across all occupations.
    """
    return find_median_hourly_wages(city_names)


def state_union_employment_skill(state_names: list[str]) -> Any:
    """
    Fetches state-level union employment rates.
    """
    return find_state_union_employment(state_names)


def state_tax_rate_skill(state_names: list[str]) -> Any:
    """
    Fetches state-level corporate income tax rates.
    """
    return find_state_tax_rate(state_names)
