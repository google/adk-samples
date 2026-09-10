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

"""Defines BigQuery product catalog connector for brand search optimization."""

import logging

from google.cloud import bigquery
from pydantic import BaseModel, Field

from ..shared_libraries import constants

logger = logging.getLogger(__name__)

client: bigquery.Client | None = None
_client_state: dict[str, BaseException | None] = {"init_error": None}


def _get_client() -> bigquery.Client | None:
    """Initializes BigQuery client lazily to avoid import-time credential errors."""
    global client
    if client is not None:
        return client
    if _client_state["init_error"] is not None:
        return None
    try:
        client = bigquery.Client()
        return client
    except Exception as e:
        logger.warning("Failed to initialize BigQuery client: %s", e)
        _client_state["init_error"] = e
        return None


class ProductRecord(BaseModel):
    """Product catalog record from BigQuery."""

    title: str = Field(description="Product title")
    description: str = Field(default="N/A", description="Product description")
    attributes: str = Field(
        default="N/A",
        description="Product attributes such as color, size, material",
    )
    brand: str = Field(description="Brand name")


class BrandCatalogResponse(BaseModel):
    """Response containing catalog products for a brand."""

    brand: str = Field(description="Requested brand name")
    products: list[ProductRecord] = Field(
        default_factory=list, description="Matching catalog products"
    )
    total_count: int = Field(
        default=0, description="Number of products retrieved"
    )


def get_product_details_for_brand(
    brand: str,
    limit: int = 5,
) -> BrandCatalogResponse:
    """Retrieves product details (title, description, attributes) for a brand from BigQuery.

    Args:
        brand: The brand name to query in the product catalog.
        limit: Maximum number of product records to retrieve (default: 5).

    Returns:
        BrandCatalogResponse containing structured product records.
    """
    if not brand or not brand.strip():
        return BrandCatalogResponse(
            brand=brand or "", products=[], total_count=0
        )

    clean_brand = brand.strip()
    bq_client = client if client is not None else _get_client()
    if bq_client is None:
        # Fallback sample data if BigQuery is unavailable in local testing/dev environments
        return BrandCatalogResponse(
            brand=clean_brand,
            products=[
                ProductRecord(
                    title=f"{clean_brand} Pro Runner",
                    description=(
                        "Comfortable and supportive performance running shoes"
                        " for active athletes. Breathable mesh upper."
                    ),
                    attributes="Size: 10, Color: Blue/Green",
                    brand=clean_brand,
                ),
                ProductRecord(
                    title=f"{clean_brand} Sportswear Graphic Tee",
                    description=("100% organic cotton daily crewneck t-shirt."),
                    attributes="Size: L, Color: Heather Black",
                    brand=clean_brand,
                ),
            ],
            total_count=2,
        )

    query = f"""
        SELECT
            Title,
            Description,
            Attributes,
            Brand
        FROM
            `{constants.PROJECT}.{constants.DATASET_ID}.{constants.TABLE_ID}`
        WHERE LOWER(Brand) LIKE LOWER(CONCAT('%', @brand_param, '%'))
        LIMIT @limit_param
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("brand_param", "STRING", clean_brand),
            bigquery.ScalarQueryParameter("limit_param", "INT64", limit),
        ]
    )

    try:
        query_job = bq_client.query(query, job_config=job_config)
        results = query_job.result()
        products = []
        for row in results:
            products.append(
                ProductRecord(
                    title=getattr(row, "Title", "")
                    or getattr(row, "title", ""),
                    description=getattr(row, "Description", "")
                    or getattr(row, "description", "")
                    or "N/A",
                    attributes=getattr(row, "Attributes", "")
                    or getattr(row, "attributes", "")
                    or "N/A",
                    brand=getattr(row, "Brand", "")
                    or getattr(row, "brand", "")
                    or clean_brand,
                )
            )
        return BrandCatalogResponse(
            brand=clean_brand,
            products=products,
            total_count=len(products),
        )
    except Exception as e:
        logger.error("BigQuery query failed: %s", e)
        return BrandCatalogResponse(
            brand=clean_brand,
            products=[],
            total_count=0,
        )
