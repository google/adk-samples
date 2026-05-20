#!/usr/bin/env python3
"""Ingest product catalog data into BigQuery.

Supports CSV and JSON source formats from GCS or local files.
Validates products against the configured schema before loading.

Usage:
    python ingest_bigquery.py \
        --project-id my-project \
        --gcs-bucket my-project-products \
        --gcs-path products.csv

    python ingest_bigquery.py \
        --project-id my-project \
        --local-file data/products.json \
        --format json

    # Or use design-spec.md for defaults:
    python ingest_bigquery.py --config design-spec.md --local-file data/products.csv
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from google.cloud import bigquery, storage

# Add _shared to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "_shared"))
from setup_utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure these for your product schema.
# Basic: product_id, name, price, description
# Standard: + category, brand, image_url
# Extended: + rating, stock, manufacturer
# Full: + variants, tags, specifications, reviews
REQUIRED_FIELDS = ["product_id", "name", "price", "description"]
OPTIONAL_FIELDS = ["category", "brand", "image_url", "rating", "stock"]

SCHEMA = [
    bigquery.SchemaField("product_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("price", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("description", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("category", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("brand", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("image_url", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("rating", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("stock", "INT64", mode="NULLABLE"),
]


def validate_product(product: dict, row_num: int) -> list[str]:
    """Validate a single product record. Returns list of error messages."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in product or not product[field]:
            errors.append(f"Row {row_num}: missing required field '{field}'")

    if "price" in product and product["price"]:
        try:
            float(product["price"])
        except (ValueError, TypeError):
            errors.append(f"Row {row_num}: 'price' must be numeric, got '{product['price']}'")

    if "stock" in product and product["stock"]:
        try:
            int(product["stock"])
        except (ValueError, TypeError):
            errors.append(f"Row {row_num}: 'stock' must be an integer, got '{product['stock']}'")

    return errors


def convert_types(product: dict) -> dict:
    """Convert string values to proper types for BigQuery."""
    converted = {}
    converted["product_id"] = product.get("product_id", "")
    converted["name"] = product.get("name", "")
    converted["description"] = product.get("description", "")

    if "price" in product and product["price"]:
        converted["price"] = float(product["price"])

    if "rating" in product and product["rating"]:
        converted["rating"] = float(product["rating"])

    if "stock" in product and product["stock"]:
        converted["stock"] = int(product["stock"])

    for field in ["category", "brand", "image_url"]:
        if field in product:
            converted[field] = product[field]

    return converted


def load_from_csv(source: str) -> list[dict]:
    """Load products from CSV (GCS URI or local path)."""
    if source.startswith("gs://"):
        parts = source.replace("gs://", "").split("/", 1)
        client = storage.Client()
        blob = client.bucket(parts[0]).blob(parts[1])
        content = blob.download_as_text()
        reader = csv.DictReader(content.splitlines())
    else:
        with open(source) as f:
            reader = csv.DictReader(f)
            return _validate_and_convert(list(reader))

    return _validate_and_convert(list(reader))


def load_from_json(source: str) -> list[dict]:
    """Load products from JSON/JSONL (GCS URI or local path)."""
    if source.startswith("gs://"):
        parts = source.replace("gs://", "").split("/", 1)
        client = storage.Client()
        blob = client.bucket(parts[0]).blob(parts[1])
        content = blob.download_as_text()
    else:
        content = Path(source).read_text()

    if source.endswith(".jsonl"):
        raw = [json.loads(line) for line in content.strip().splitlines() if line.strip()]
    else:
        parsed = json.loads(content)
        raw = parsed if isinstance(parsed, list) else parsed.get("products", [])

    return _validate_and_convert(raw)


def _validate_and_convert(raw_products: list[dict]) -> list[dict]:
    """Validate and type-convert a list of product dicts."""
    products = []
    all_errors = []

    for i, product in enumerate(raw_products, start=1):
        errors = validate_product(product, i)
        if errors:
            all_errors.extend(errors)
            continue
        products.append(convert_types(product))

    if all_errors:
        logger.warning(f"Skipped rows with {len(all_errors)} validation errors:")
        for err in all_errors[:10]:
            logger.warning(f"  {err}")
        if len(all_errors) > 10:
            logger.warning(f"  ... and {len(all_errors) - 10} more")

    logger.info(f"Loaded {len(products)} valid products")
    return products


def ingest(
    project_id: str,
    dataset_id: str,
    table_id: str,
    source: str,
    source_format: str = "csv",
):
    """Ingest products into BigQuery."""
    client = bigquery.Client(project=project_id)

    # Check if dataset already exists. Never modify or delete existing datasets.
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        client.get_dataset(dataset_ref)
        if sys.stdin.isatty():
            # Interactive terminal -- ask the user for a new name
            logger.warning(f"Dataset {dataset_ref} already exists.")
            try:
                new_name = input("  Enter a different dataset name (or Ctrl+C to cancel): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                logger.info("Cancelled.")
                sys.exit(0)
            if not new_name:
                logger.error("No name provided. Exiting.")
                sys.exit(1)
            dataset_id = new_name
            dataset_ref = f"{project_id}.{dataset_id}"
            try:
                client.get_dataset(dataset_ref)
                logger.error(f"Dataset {dataset_ref} also exists. Re-run with --dataset-id <unique_name>.")
                sys.exit(1)
            except Exception:
                pass  # New name is available
        else:
            # Non-interactive (agent/subprocess) -- fail with a clear message
            logger.error(
                f"Dataset {dataset_ref} already exists. "
                f"Re-run with a different name: --dataset-id {dataset_id}_v2"
            )
            sys.exit(1)
    except Exception:
        pass  # Dataset doesn't exist, safe to create

    logger.info(f"Creating dataset {dataset_ref}")
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    client.create_dataset(dataset)

    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    logger.info(f"Creating table {table_ref}")
    table = bigquery.Table(table_ref, schema=SCHEMA)
    client.create_table(table)

    # Load products
    if source_format == "json":
        products = load_from_json(source)
    else:
        products = load_from_csv(source)

    if not products:
        logger.error("No valid products to ingest")
        sys.exit(1)

    errors = client.insert_rows_json(table_ref, products)
    if errors:
        logger.error(f"Errors inserting rows: {errors}")
        sys.exit(1)

    logger.info(f"Successfully ingested {len(products)} products to {table_ref}")


def main():
    parser = argparse.ArgumentParser(description="Ingest product catalog to BigQuery")
    parser.add_argument("--config", default="", help="Path to design-spec.md (provides defaults for other args)")
    parser.add_argument("--project-id", help="GCP project ID")
    parser.add_argument("--dataset-id", default="retail_skill_products", help="BigQuery dataset ID")
    parser.add_argument("--table-id", default="products", help="BigQuery table ID")
    parser.add_argument("--gcs-bucket", help="GCS bucket name (used with --gcs-path)")
    parser.add_argument("--gcs-path", help="Path to data file in GCS bucket")
    parser.add_argument("--local-file", help="Path to local data file")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Source format")

    args = parser.parse_args()

    # Load design-spec.md defaults -- CLI args override config values
    if args.config:
        cfg = load_config(args.config)
        if not args.project_id:
            args.project_id = cfg.get("gcp_project_id", "")

    if not args.project_id:
        parser.error("--project-id is required (or set gcp_project_id in design-spec.md)")

    if args.local_file:
        source = args.local_file
    elif args.gcs_bucket and args.gcs_path:
        source = f"gs://{args.gcs_bucket}/{args.gcs_path}"
    else:
        parser.error("Provide either --local-file or both --gcs-bucket and --gcs-path")

    ingest(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        table_id=args.table_id,
        source=source,
        source_format=args.format,
    )


if __name__ == "__main__":
    main()
