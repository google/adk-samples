# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines tools for brand search optimization agent"""

from google.cloud import bigquery
from google.adk.tools import ToolContext

from ..shared_libraries import constants

# Initialize the BigQuery client outside the function
try:
    client = bigquery.Client()  # Initialize client once
except Exception as e:
    print(f"Error initializing BigQuery client: {e}")
    client = None  # Set client to None if initialization fails


def get_product_details_for_brand(tool_context: ToolContext):
    """
    Retrieves product details (title, description, attributes, and brand) from a BigQuery table for a tool_context.

    Args:
        tool_context (str): The tool_context to search for (using a LIKE '%brand%' query).

    Returns:
        str: A markdown table containing the product details, or an error message if BigQuery client initialization failed.
             The table includes columns for 'Title', 'Description', 'Attributes', and 'Brand'.
             Returns a maximum of 3 results.

    Example:
        >>> get_product_details_for_brand(tool_context)
        '| Title | Description | Attributes | Brand |\\n|---|---|---|---|\\n| Nike Air Max | Comfortable running shoes | Size: 10, Color: Blue | Nike\\n| Nike Sportswear T-Shirt | Cotton blend, short sleeve | Size: L, Color: Black | Nike\\n| Nike Pro Training Shorts | Moisture-wicking fabric | Size: M, Color: Gray | Nike\\n'
    """
    brand = tool_context.user_content.parts[0].text
    if client is None:  # Check if client initialization failed
        return "BigQuery client initialization failed. Cannot execute query."

    query = f"""
        SELECT
            Title,
            Description,
            Attributes,
            Brand
        FROM
            {constants.PROJECT}.{constants.DATASET_ID}.{constants.TABLE_ID}
        WHERE brand LIKE '%{brand}%'
        LIMIT 3
    """
    query_job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("parameter1", "STRING", brand)
        ]
    )

    query_job = client.query(query, job_config=query_job_config)
    query_job = client.query(query)
    results = query_job.result()

    markdown_table = "| Title | Description | Attributes | Brand |\n"
    markdown_table += "|---|---|---|---|\n"

    for row in results:
        title = row.Title
        description = row.Description if row.Description else "N/A"
        attributes = row.Attributes if row.Attributes else "N/A"

        markdown_table += (
            f"| {title} | {description} | {attributes} | {brand}\n"
        )

    return markdown_table
