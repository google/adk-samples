#  Copyright 2025 Google LLC. This software is provided as-is, without warranty
#  or representation for any use or purpose. Your use of it is subject to your
#  agreement with Google.
"""ADK Skill: Economic Visualization (Plotly). Hardened charts for executive reporting."""

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.io as pio
from pydantic import BaseModel, Field


class VisualizationRequest(BaseModel):
    data: list[dict[str, Any]] = Field(
        ..., description="List of dictionaries containing the economic data."
    )
    x_axis: str = Field(..., description="Column name for the X-axis.")
    y_axis: str = Field(..., description="Column name for the Y-axis.")
    chart_type: str = Field(
        "bar", description="Type of chart to generate (bar, line, scatter)."
    )
    title: str | None = Field(None, description="Title of the chart.")
    color_by: str | None = Field(
        None, description="Optional column name to color the data points by."
    )


def generate_economic_chart(
    data: list[dict[str, Any]],
    x_axis: str,
    y_axis: str,
    chart_type: str = "bar",
    title: str | None = None,
    color_by: str | None = None,
) -> str:
    """
    Generates a Plotly JSON string for an economic chart based on the provided data.
    Use this tool when you need to provide a visual ROI matrix or trend analysis to a senior stakeholder.
    """
    if not data:
        return "ERROR: No data provided for chart generation."

    df = pd.DataFrame(data)

    if x_axis not in df.columns or y_axis not in df.columns:
        return f"ERROR: Columns '{x_axis}' or '{y_axis}' not found in data. Available: {list(df.columns)}"

    # Ensure numeric columns are actually numeric
    try:
        df[y_axis] = pd.to_numeric(
            df[y_axis].replace(r"[\$,%]", "", regex=True)
        )
    except (ValueError, TypeError):
        pass

    fig = None
    if chart_type.lower() == "bar":
        fig = px.bar(
            df,
            x=x_axis,
            y=y_axis,
            title=title,
            color=color_by,
            template="plotly_white",
        )
    elif chart_type.lower() == "line":
        fig = px.line(
            df,
            x=x_axis,
            y=y_axis,
            title=title,
            color=color_by,
            template="plotly_white",
        )
    elif chart_type.lower() == "scatter":
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            title=title,
            color=color_by,
            template="plotly_white",
        )
    else:
        fig = px.bar(
            df,
            x=x_axis,
            y=y_axis,
            title=title,
            color=color_by,
            template="plotly_white",
        )

    # High-fidelity styling
    fig.update_layout(
        font_family="Roboto, sans-serif",
        xaxis_tickangle=-45,
        margin={"l": 20, "r": 20, "t": 50, "b": 100},
    )

    # Return as JSON string for frontend rendering
    return pio.to_json(fig)
