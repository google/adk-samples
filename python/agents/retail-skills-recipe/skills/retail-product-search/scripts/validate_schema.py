#!/usr/bin/env python3
"""Validate a product catalog CSV/JSON against the expected schema.

Usage:
    python validate_schema.py --file products.csv --fields-level Standard
    python validate_schema.py --file products.json --fields-level Extended
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIELD_LEVELS = {
    "Basic": {
        "required": ["product_id", "name", "price", "description"],
        "optional": [],
    },
    "Standard": {
        "required": ["product_id", "name", "price", "description"],
        "optional": ["category", "brand", "image_url"],
    },
    "Extended": {
        "required": ["product_id", "name", "price", "description"],
        "optional": ["category", "brand", "image_url", "rating", "stock", "manufacturer"],
    },
    "Full": {
        "required": ["product_id", "name", "price", "description"],
        "optional": [
            "category", "brand", "image_url", "rating", "stock",
            "manufacturer", "variants", "tags", "specifications", "reviews",
        ],
    },
}


def load_records(file_path: Path) -> list[dict]:
    """Load records from CSV or JSON file."""
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        with open(file_path) as f:
            return list(csv.DictReader(f))
    elif suffix in (".json", ".jsonl"):
        content = file_path.read_text()
        if suffix == ".jsonl":
            return [json.loads(line) for line in content.strip().splitlines() if line.strip()]
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "products" in parsed:
            return parsed["products"]
        raise ValueError("JSON must be an array or {\"products\": [...]}")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .json, or .jsonl")


def validate(records: list[dict], fields_level: str) -> tuple[int, list[str]]:
    """Validate records against the schema. Returns (valid_count, errors)."""
    schema = FIELD_LEVELS[fields_level]
    required = schema["required"]
    all_fields = set(required + schema["optional"])
    errors = []
    valid = 0

    for i, record in enumerate(records, start=1):
        row_errors = []

        for field in required:
            if field not in record or not record[field]:
                row_errors.append(f"Row {i}: missing required field '{field}'")

        if "price" in record and record["price"]:
            try:
                float(record["price"])
            except (ValueError, TypeError):
                row_errors.append(f"Row {i}: 'price' must be numeric, got '{record['price']}'")

        if "rating" in record and record["rating"]:
            try:
                val = float(record["rating"])
                if not (0 <= val <= 5):
                    row_errors.append(f"Row {i}: 'rating' should be 0-5, got {val}")
            except (ValueError, TypeError):
                row_errors.append(f"Row {i}: 'rating' must be numeric")

        if "stock" in record and record["stock"]:
            try:
                int(record["stock"])
            except (ValueError, TypeError):
                row_errors.append(f"Row {i}: 'stock' must be an integer")

        extra_fields = set(record.keys()) - all_fields
        if extra_fields:
            row_errors.append(f"Row {i}: unexpected fields: {extra_fields}")

        if row_errors:
            errors.extend(row_errors)
        else:
            valid += 1

    return valid, errors


def main():
    parser = argparse.ArgumentParser(description="Validate product catalog schema")
    parser.add_argument("--file", required=True, help="Path to product data file (CSV/JSON)")
    parser.add_argument(
        "--fields-level",
        choices=["Basic", "Standard", "Extended", "Full"],
        default="Standard",
        help="Product fields level (default: Standard)",
    )

    args = parser.parse_args()
    file_path = Path(args.file)

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    records = load_records(file_path)
    logger.info(f"Loaded {len(records)} records from {file_path}")

    valid_count, errors = validate(records, args.fields_level)

    if errors:
        logger.warning(f"{len(errors)} validation errors found:")
        for err in errors[:20]:
            logger.warning(f"  {err}")
        if len(errors) > 20:
            logger.warning(f"  ... and {len(errors) - 20} more")

    logger.info(f"Validation complete: {valid_count}/{len(records)} records valid")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
