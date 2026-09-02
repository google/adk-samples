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

"""Defines BigQuery catalog connector tool for Brand Search Optimization."""

import os

from google.cloud import bigquery

client = None
_client_state = {"init_error": None}


def _get_client():
    """Initializes a BigQuery client on demand to avoid import-time failures."""
    if client is not None:
        return client
    if _client_state["init_error"] is not None:
        return None
    try:
        return bigquery.Client()
    except Exception as e:
        print(f"Error initializing BigQuery client: {e}")
        _client_state["init_error"] = e
        return None


def get_product_details_for_brand(brand_name: str) -> str:
    """Retrieves product details (title, description, attributes, and brand) from a BigQuery catalog table.

    Args:
        brand_name: The name of the brand to search for (e.g. 'BSOAgentTestBrand', 'Nike', 'Adidas').

    Returns:
        str: A markdown table containing the product details or an informative error message.
    """
    brand = brand_name.strip()
    if not brand:
        return "Brand name cannot be empty. Please provide a brand name."

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        return "Error: GOOGLE_CLOUD_PROJECT environment variable is not set."

    dataset_id = os.getenv("DATASET_ID", "products_data_agent").strip()
    table_id = os.getenv("TABLE_ID", "shoe_items").strip()

    bq_client = client if client is not None else _get_client()
    if bq_client is None:
        return "BigQuery client initialization failed. Please check your credentials."

    query = f"""
        SELECT
            Title,
            Description,
            Attributes,
            Brand
        FROM
            `{project}.{dataset_id}.{table_id}`
        WHERE LOWER(Brand) LIKE CONCAT('%', LOWER(@parameter1), '%')
        LIMIT 5
    """
    query_job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("parameter1", "STRING", brand)
        ]
    )

    try:
        query_job = bq_client.query(query, job_config=query_job_config)
        results = query_job.result()
    except Exception as e:
        return f"Error querying BigQuery catalog: {e}"

    markdown_table = "| Title | Description | Attributes | Brand |\n"
    markdown_table += "|---|---|---|---|\n"

    row_count = 0
    for row in results:
        row_count += 1
        title = row.Title if getattr(row, "Title", None) else "N/A"
        description = (
            row.Description if getattr(row, "Description", None) else "N/A"
        )
        attributes = (
            row.Attributes if getattr(row, "Attributes", None) else "N/A"
        )
        row_brand = row.Brand if getattr(row, "Brand", None) else brand

        markdown_table += (
            f"| {title} | {description} | {attributes} | {row_brand} |\n"
        )

    if row_count == 0:
        return f"No product catalog entries found for brand '{brand}'."

    return markdown_table
