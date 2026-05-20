#!/usr/bin/env python3
"""Export generated product content from BigQuery to CSV or JSON.

Usage:
    python export_content.py --config design-spec.md --format csv --output products_content.csv

    python export_content.py \
        --project-id my-project \
        --format json \
        --output content.json \
        --type description
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from google.cloud import bigquery

# Add _shared to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "_shared"))
from setup_utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export(
    project_id: str,
    dataset_id: str,
    output_path: str,
    output_format: str,
    content_type: str = "",
    language: str = "",
) -> None:
    """Export generated content from BigQuery."""
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.content_generated"

    conditions = []
    query_params = []
    if content_type:
        conditions.append("content_type = @content_type")
        query_params.append(bigquery.ScalarQueryParameter("content_type", "STRING", content_type))
    if language:
        conditions.append("language = @language")
        query_params.append(bigquery.ScalarQueryParameter("language", "STRING", language))

    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT * FROM `{table_ref}`{where} ORDER BY product_id, content_type, variant"

    logger.info(f"Querying: {query}")
    job_config = bigquery.QueryJobConfig(query_parameters=query_params) if query_params else None
    results = client.query(query, job_config=job_config).result()
    rows = [dict(row.items()) for row in results]

    if not rows:
        logger.warning("No content found to export")
        return

    path = Path(output_path)

    if output_format == "csv":
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text(json.dumps(rows, indent=2, default=str))

    logger.info(f"Exported {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Export generated product content")
    parser.add_argument("--config", default="", help="Path to design-spec.md")
    parser.add_argument("--project-id", help="GCP project ID")
    parser.add_argument("--dataset-id", default="retail_skill_products", help="BigQuery dataset")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--type", default="", help="Filter by content type")
    parser.add_argument("--language", default="", help="Filter by language")

    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        if not args.project_id:
            args.project_id = cfg.get("gcp_project_id", "")

    if not args.project_id:
        parser.error("--project-id is required (or set gcp_project_id in design-spec.md)")

    export(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        output_path=args.output,
        output_format=args.format,
        content_type=args.type,
        language=args.language,
    )


if __name__ == "__main__":
    main()
