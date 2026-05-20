#!/usr/bin/env python3
"""Ingest user behavior events into BigQuery for product recommendations.

Loads user interaction data (product views, add-to-cart, purchases) into
BigQuery. This data powers collaborative filtering and Vertex AI
Recommendations AI.

Usage:
    python ingest_user_events.py \
        --project-id my-project \
        --gcs-bucket my-project-data \
        --gcs-path events/user_events.csv

    python ingest_user_events.py \
        --project-id my-project \
        --local-file data/user_events.jsonl \
        --format json

    # Using design-spec.md for defaults
    python ingest_user_events.py \
        --config design-spec.md \
        --local-file data/user_events.csv
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

REQUIRED_FIELDS = ["event_id", "user_id", "product_id", "event_type", "timestamp"]
OPTIONAL_FIELDS = ["session_id", "event_metadata"]

VALID_EVENT_TYPES = {
    "detail-page-view",
    "add-to-cart",
    "purchase",
    "search",
    "add-to-wishlist",
    "review",
}

SCHEMA = [
    bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("product_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("event_metadata", "JSON", mode="NULLABLE"),
]


def validate_event(event: dict, row_num: int) -> list[str]:
    """Validate a single user event record."""
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in event or not event[field]:
            errors.append(f"Row {row_num}: missing required field '{field}'")

    event_type = event.get("event_type", "")
    if event_type and event_type not in VALID_EVENT_TYPES:
        errors.append(
            f"Row {row_num}: invalid event_type '{event_type}', "
            f"must be one of: {', '.join(sorted(VALID_EVENT_TYPES))}"
        )

    return errors


def load_from_csv(source: str) -> list[dict]:
    """Load user events from CSV (GCS URI or local path)."""
    if source.startswith("gs://"):
        parts = source.replace("gs://", "").split("/", 1)
        client = storage.Client()
        blob = client.bucket(parts[0]).blob(parts[1])
        content = blob.download_as_text()
        reader = csv.DictReader(content.splitlines())
        rows = list(reader)
    else:
        with open(source) as f:
            rows = list(csv.DictReader(f))

    return _validate_and_convert(rows)


def load_from_json(source: str) -> list[dict]:
    """Load user events from JSON/JSONL (GCS URI or local path)."""
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
        raw = parsed if isinstance(parsed, list) else parsed.get("events", [])

    return _validate_and_convert(raw)


def _validate_and_convert(raw_events: list[dict]) -> list[dict]:
    """Validate and normalize a list of event dicts."""
    events = []
    all_errors = []

    for i, event in enumerate(raw_events, start=1):
        errors = validate_event(event, i)
        if errors:
            all_errors.extend(errors)
            continue

        converted = {
            "event_id": event["event_id"],
            "user_id": event["user_id"],
            "product_id": event["product_id"],
            "event_type": event["event_type"],
            "timestamp": event["timestamp"],
        }

        if event.get("session_id"):
            converted["session_id"] = event["session_id"]

        metadata = event.get("event_metadata", "")
        if metadata:
            if isinstance(metadata, str):
                converted["event_metadata"] = metadata
            else:
                converted["event_metadata"] = json.dumps(metadata)

        events.append(converted)

    if all_errors:
        logger.warning(f"Skipped {len(all_errors)} rows with validation errors:")
        for err in all_errors[:10]:
            logger.warning(f"  {err}")
        if len(all_errors) > 10:
            logger.warning(f"  ... and {len(all_errors) - 10} more")

    logger.info(f"Loaded {len(events)} valid events")
    return events


def ingest(
    project_id: str,
    dataset_id: str,
    table_id: str,
    source: str,
    source_format: str = "csv",
):
    """Ingest user events into BigQuery."""
    client = bigquery.Client(project=project_id)

    # Check if dataset already exists. Never modify or delete existing datasets.
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        client.get_dataset(dataset_ref)
        if sys.stdin.isatty():
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
                pass
        else:
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

    # Create table
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    try:
        logger.info(f"Creating table {table_ref}")
        table = bigquery.Table(table_ref, schema=SCHEMA)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp",
        )
        client.create_table(table)
    except Exception:
        pass  # Table already exists, safe to proceed

    # Load events
    if source_format == "json":
        events = load_from_json(source)
    else:
        events = load_from_csv(source)

    if not events:
        logger.error("No valid events to ingest")
        sys.exit(1)

    errors = client.insert_rows_json(table_ref, events)
    if errors:
        logger.error(f"Errors inserting rows: {errors}")
        sys.exit(1)

    logger.info(f"Successfully ingested {len(events)} user events to {table_ref}")


def main():
    parser = argparse.ArgumentParser(description="Ingest user behavior events to BigQuery")
    parser.add_argument("--config", default="", help="Path to design-spec.md (provides defaults for other args)")
    parser.add_argument("--project-id", help="GCP project ID")
    parser.add_argument("--dataset-id", default="retail_skill_products", help="BigQuery dataset ID")
    parser.add_argument("--table-id", default="user_events", help="BigQuery table ID")
    parser.add_argument("--gcs-bucket", help="GCS bucket name (used with --gcs-path)")
    parser.add_argument("--gcs-path", help="Path to events file in GCS bucket")
    parser.add_argument("--local-file", help="Path to local events file")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Source format")

    args = parser.parse_args()

    # Load design-spec.md defaults
    if args.config:
        cfg = load_config(args.config)
        if not args.project_id:
            args.project_id = cfg.get("gcp_project_id", "")
        if args.dataset_id == "retail_skill_products" and cfg.get("dataset_id"):
            args.dataset_id = cfg["dataset_id"]
        if args.table_id == "user_events" and cfg.get("user_events_table"):
            args.table_id = cfg["user_events_table"]

        # Check if user events are needed for the recommendation type
        rec_type = cfg.get("recommendation_type", "collaborative")
        has_events = cfg.get("has_user_events", "Yes")
        if rec_type == "content-based" and has_events == "No":
            logger.info("Content-based recommendations selected with no user events.")
            logger.info("Skipping user event ingestion (not required for content-based).")
            sys.exit(0)

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
