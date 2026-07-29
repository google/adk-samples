"""Visualization tools for the YouTube Analyst agent."""

import io
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.adk.agents.context import Context as ToolContext
from google.genai import types

logger = logging.getLogger(__name__)


async def execute_visualization_code(
    code: str,
    tool_context: ToolContext,
    chart_type: str = "plotly",
    filename: str = "chart.html",
) -> str:
    """
    Executes Python code to generate a chart and saves it as an artifact.
    The code should assume that 'pd', 'px', 'go', and 'plt' are already imported.
    The code MUST assign the resulting chart object to a variable named 'fig'.

    Args:
        code: The Python code to execute.
        tool_context: The ADK tool context.
        chart_type: The library used ('plotly' or 'matplotlib').
        filename: The filename for the saved artifact.

    Returns:
        A message confirming the artifact creation and its ID.
    """
    try:
        # Prepare the execution environment
        local_vars: dict[str, Any] = {
            "pd": pd,
            "px": px,
            "go": go,
            "plt": plt,
            "fig": None,
        }

        # Execute the code
        # WARNING: executing arbitrary code is dangerous in production.
        # This is a sample intended for demonstration purposes.
        exec(code, {}, local_vars)

        fig = local_vars.get("fig")
        if fig is None:
            return "ERROR: The code did not assign a chart object to the 'fig' variable."

        # Save the result as an artifact
        if chart_type == "plotly":
            # Plotly figures can be converted to JSON or HTML
            chart_html = fig.to_html(include_plotlyjs="cdn", full_html=False)  # type: ignore
            artifact_data = chart_html.encode("utf-8")
            mime_type = "text/html"
        else:
            # Matplotlib figures can be saved to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            artifact_data = buf.getvalue()
            mime_type = "image/png"

        # Save to file system for local preview
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(artifact_data)

        # Register as ADK artifact
        artifact_id = filename.replace(".", "_")
        await tool_context.save_artifact(
            artifact_id,
            types.Part.from_bytes(data=artifact_data, mime_type=mime_type),
        )

        return f"Successfully generated {chart_type} chart. Artifact ID: {artifact_id}"

    except Exception as e:
        logger.error(f"Failed to execute visualization code: {e}")
        return f"ERROR: {e!s}"


def get_visualization_instructions() -> str:
    """
    Returns instructions for the visualization agent on how to use the available tools.
    """
    return """
    You are an expert at creating interactive data visualizations using Python (Plotly and Matplotlib).
    Your goal is to turn YouTube data into clear, insightful charts.

    ### 🛠️ Available Libraries:
    - `pandas` (as `pd`)
    - `plotly.express` (as `px`)
    - `plotly.graph_objects` (as `go`)
    - `matplotlib.pyplot` (as `plt`)

    ### 📏 Mandatory Rule:
    The code you generate MUST assign the final chart object to a variable named `fig`.

    ### 🎨 Styling Guidelines:
    - Use clear titles and axis labels.
    - For Plotly, use the 'plotly_white' template for a clean look.
    - Ensure charts are responsive and readable.

    ### 📝 Example (Plotly):
    ```python
    data = {"Video": ["A", "B", "C"], "Views": [100, 200, 150]}
    df = pd.DataFrame(data)
    fig = px.bar(df, x="Video", y="Views", title="Video Performance")
    fig.update_layout(template="plotly_white")
    ```
    """
