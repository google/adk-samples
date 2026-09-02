# Copyright 2026 Google LLC
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
CATALOG_QUERY_LIMIT = 5
DEFAULT_FIELD_PLACEHOLDER = "N/A"


def _get_client():
    """Initializes a BigQuery client on demand to avoid import-time failures."""
    global client
    if client is not None:
        return client
    if _client_state["init_error"] is not None:
        return None
    try:
        client = bigquery.Client()
        return client
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

    project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    if not project:
        return "Error: GOOGLE_CLOUD_PROJECT environment variable is not set."

    dataset_id = (os.getenv("DATASET_ID") or "").strip()
    if not dataset_id:
        return "Error: DATASET_ID environment variable is not set."

    table_id = (os.getenv("TABLE_ID") or "").strip()
    if not table_id:
        return "Error: TABLE_ID environment variable is not set."

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
        LIMIT {CATALOG_QUERY_LIMIT}
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
        title = getattr(row, "Title", None) or DEFAULT_FIELD_PLACEHOLDER
        description = (
            getattr(row, "Description", None) or DEFAULT_FIELD_PLACEHOLDER
        )
        attributes = (
            getattr(row, "Attributes", None) or DEFAULT_FIELD_PLACEHOLDER
        )
        row_brand = getattr(row, "Brand", None) or brand

        markdown_table += (
            f"| {title} | {description} | {attributes} | {row_brand} |\n"
        )

    if row_count == 0:
        return f"No product catalog entries found for brand '{brand}'."

    return markdown_table
