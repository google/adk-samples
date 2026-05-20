#!/usr/bin/env python3
"""Batch generate product content using Gemini.

Reads products from BigQuery, generates descriptions, SEO titles,
meta descriptions, marketing copy, or translations using Gemini.
Writes output to BigQuery or file.

Usage:
    python generate_content.py --config design-spec.md --type description

    python generate_content.py \
        --project-id my-project \
        --type seo \
        --model gemini-2.5-flash

    python generate_content.py \
        --config design-spec.md \
        --type translation \
        --languages es,fr,de
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from google.cloud import bigquery
from google import genai
from google.genai import types

# Add _shared to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "_shared"))
from setup_utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 50
DEFAULT_TONE = "Professional and informative"

CONTENT_TYPES = ["description", "seo_title", "meta_description", "marketing_copy", "translation"]


def fetch_products(project_id: str, dataset_id: str, table_id: str) -> List[Dict]:
    """Fetch products from BigQuery."""
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
    results = client.query(query).result()
    products = [dict(row.items()) for row in results]
    logger.info(f"Fetched {len(products)} products")
    return products


def build_prompt(product: Dict, content_type: str, config: Dict) -> str:
    """Build the Gemini prompt for content generation."""
    tone = config.get("brand_tone", DEFAULT_TONE)
    brand = config.get("brand_name", product.get("brand", ""))
    always_include = config.get("always_include", "")
    never_use = config.get("never_use", "")
    target_length = config.get("description_length", "Medium (100-150 words)")

    product_info = (
        f"Product: {product.get('name', '')}\n"
        f"Category: {product.get('category', '')}\n"
        f"Brand: {brand}\n"
        f"Price: ${product.get('price', '')}\n"
        f"Description: {product.get('description', '')}\n"
    )

    if product.get("rating"):
        product_info += f"Rating: {product['rating']}/5\n"

    constraints = f"\nTone: {tone}\n"
    if always_include:
        constraints += f"Always include: {always_include}\n"
    if never_use:
        constraints += f"Never use these words: {never_use}\n"

    if content_type == "description":
        return (
            f"Write a {target_length.split('(')[0].strip().lower()} product description "
            f"for the following product.\n\n{product_info}{constraints}\n"
            f"Include key features and benefits. "
            f"{'Include technical specifications as bullet points.' if config.get('include_specs', True) else ''} "
            f"{'Include a \"Who it is for\" section.' if config.get('include_use_cases', True) else ''}"
        )

    elif content_type == "seo_title":
        return (
            f"Write an SEO-optimized product page title (50-60 characters max) "
            f"for this product.\n\n{product_info}{constraints}\n"
            f"{'Include the price.' if config.get('include_price_in_seo', False) else 'Do NOT include the price.'}\n"
            f"Return ONLY the title, no quotes or extra text."
        )

    elif content_type == "meta_description":
        return (
            f"Write an SEO meta description (150-160 characters max) "
            f"for this product page.\n\n{product_info}{constraints}\n"
            f"Make it compelling and include a call to action. "
            f"Return ONLY the meta description."
        )

    elif content_type == "marketing_copy":
        return (
            f"Write a short marketing headline and subheadline for this product "
            f"(suitable for an ad or email).\n\n{product_info}{constraints}\n"
            f"Format:\nHeadline: ...\nSubheadline: ..."
        )

    elif content_type == "translation":
        target_lang = config.get("target_language", "es")
        return (
            f"Translate the following product description to {target_lang}. "
            f"Preserve the meaning and tone. Do NOT translate brand names.\n\n"
            f"Description: {product.get('description', '')}\n\n"
            f"Return ONLY the translated text."
        )

    return f"Describe this product:\n\n{product_info}"


def generate_for_product(
    client: genai.Client,
    model: str,
    product: Dict,
    content_type: str,
    config: Dict,
) -> str:
    """Generate content for a single product."""
    prompt = build_prompt(product, content_type, config)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7 if content_type == "marketing_copy" else 0.3,
            max_output_tokens=500,
        ),
    )

    return response.text.strip()


def save_to_bigquery(
    project_id: str,
    dataset_id: str,
    results: List[Dict],
) -> None:
    """Save generated content to BigQuery."""
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.content_generated"

    # Create table if needed
    schema = [
        bigquery.SchemaField("product_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("content_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("language", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("model", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("variant", "INT64", mode="NULLABLE"),
    ]

    try:
        client.get_table(table_id)
    except Exception:
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        logger.info(f"Created table {table_id}")

    errors = client.insert_rows_json(table_id, results)
    if errors:
        logger.error(f"Errors saving to BigQuery: {errors}")
    else:
        logger.info(f"Saved {len(results)} rows to {table_id}")


def save_to_file(results: List[Dict], output_path: str, fmt: str) -> None:
    """Save generated content to file."""
    path = Path(output_path)

    if fmt == "json":
        path.write_text(json.dumps(results, indent=2))
    elif fmt == "csv":
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    logger.info(f"Saved {len(results)} results to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate product content with Gemini")
    parser.add_argument("--config", default="", help="Path to design-spec.md")
    parser.add_argument("--project-id", help="GCP project ID")
    parser.add_argument("--dataset-id", default="retail_skill_products", help="BigQuery dataset")
    parser.add_argument("--table-id", default="products", help="BigQuery products table")
    parser.add_argument("--type", choices=CONTENT_TYPES, default=None, help="Content type to generate (or set content_type in design-spec.md)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Products per batch")
    parser.add_argument("--languages", default="", help="Target languages for translation (comma-separated)")
    parser.add_argument("--variants", type=int, default=1, help="Number of A/B variants")
    parser.add_argument("--output", default="", help="Output file path (default: BigQuery)")
    parser.add_argument("--output-format", choices=["json", "csv"], default="json", help="Output file format")

    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)
        if not args.project_id:
            args.project_id = config.get("gcp_project_id", "")
        # Read content settings from design-spec if not provided via CLI
        if not hasattr(args, 'type') or args.type is None:
            args.type = config.get("content_type", "descriptions")
        if args.variants == 1 and config.get("ab_variants"):
            args.variants = int(config["ab_variants"])
        if not args.languages:
            target_langs = config.get("target_languages", [])
            if target_langs:
                args.languages = ",".join(target_langs)
        if config.get("brand_tone"):
            args.tone = config["brand_tone"]
        if config.get("gemini_model"):
            args.model = config["gemini_model"]
        if config.get("batch_size"):
            args.batch_size = int(config["batch_size"])

    if not args.project_id:
        parser.error("--project-id is required (or set gcp_project_id in design-spec.md)")

    # Initialize Gemini client
    import os
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    gemini_client = genai.Client(vertexai=True, project=args.project_id, location="us-central1")

    # Fetch products
    products = fetch_products(args.project_id, args.dataset_id, args.table_id)
    if not products:
        logger.error("No products found")
        sys.exit(1)

    # Generate content
    all_results = []
    languages = [l.strip() for l in args.languages.split(",") if l.strip()] if args.languages else [""]

    for i, product in enumerate(products):
        product_id = str(product.get("product_id", ""))

        for lang in languages:
            lang_config = dict(config)
            if lang:
                lang_config["target_language"] = lang

            for variant in range(args.variants):
                try:
                    content = generate_for_product(
                        gemini_client, args.model, product, args.type, lang_config,
                    )

                    all_results.append({
                        "product_id": product_id,
                        "content_type": args.type,
                        "content": content,
                        "language": lang or "en",
                        "model": args.model,
                        "variant": variant + 1,
                    })

                except Exception as e:
                    logger.warning(f"Failed to generate for {product_id}: {e}")

        if (i + 1) % 10 == 0:
            logger.info(f"Generated content for {i + 1}/{len(products)} products")
            time.sleep(1)  # Rate limiting

    logger.info(f"Generated {len(all_results)} content pieces")

    # Save results
    if args.output:
        save_to_file(all_results, args.output, args.output_format)
    else:
        save_to_bigquery(args.project_id, args.dataset_id, all_results)


if __name__ == "__main__":
    main()
