#  Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""Pydantic models for Economic Research Agent (ERA)."""

from pydantic import BaseModel


class MetroMatrix(BaseModel):
    """Metro Matrix workflow object."""

    city: str | None = None
    state: str | None = None
    county: str | None = None


class HQRelocation(BaseModel):
    """Head Quarter Relocation workflow object."""

    city: str | None = None
    state: str | None = None
    county: str | None = None
    industry: str | None = None


class CompanyRelocation(BaseModel):
    """Company Relocation workflow object."""

    city: str | None = None
    state: str | None = None
    county: str | None = None
    industry: str | None = None


class MetroMatrixResult(BaseModel):
    """Metro Matrix analysis result object."""

    city_analysis: list[MetroMatrix] = []
    error: str | None = None


class HQRelocationResult(BaseModel):
    """Head Quarter Relocation analysis result object."""

    city_analysis: list[HQRelocation] = []
    error: str | None = None


class CompanyRelocationResult(BaseModel):
    """Company Relocation Result analysist result object."""

    city_analysis: list[CompanyRelocation] = []
    error: str | None = None
