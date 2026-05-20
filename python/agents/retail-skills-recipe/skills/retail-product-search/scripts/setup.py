#!/usr/bin/env python3
"""Automated project setup driven by design-spec.md.

Reads the design-spec.md configuration and runs the appropriate scripts
based on the user's answers. This is the single entry point for both
the `vs` CLI and manual setup.

Usage:
    python scripts/setup.py --config assets/design-spec.md
    python scripts/setup.py --config assets/design-spec.md --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

# Add skills/ to sys.path so _shared/ is importable as a top-level package.
# Do NOT add an __init__.py to skills/ -- that would break this import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _shared.setup_utils import load_config, run_step

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def setup(config_path: str, dry_run: bool = False):
    """Run the full setup pipeline based on design-spec.md."""
    cfg = load_config(config_path)

    project_id = cfg.get("gcp_project_id", "")
    if not project_id:
        logger.error("gcp_project_id is required in design-spec.md")
        sys.exit(1)

    has_existing_api = cfg.get("has_existing_api", "No")
    has_images = cfg.get("has_images", "No images")
    voice = cfg.get("voice_capabilities", "No")
    product_fields = cfg.get("product_fields", "Extended")
    dataset_id = cfg.get("dataset_id", "retail_skill_products")
    table_id = cfg.get("table_id", "products")
    catalog_size = cfg.get("catalog_size", "1K-50K")
    search_type = cfg.get("search_type", "Text-only")
    vague_query_handling = cfg.get("vague_query_handling", "Ask 1-2 clarifying questions")
    ui_type = cfg.get("user_interface", "Cloud Run web app")
    gcp_region = cfg.get("gcp_region", "us-central1")

    logger.info("=" * 60)
    logger.info("SETUP PIPELINE (driven by design-spec.md)")
    logger.info("=" * 60)
    logger.info(f"  Project:      {project_id}")
    logger.info(f"  Region:       {gcp_region}")
    logger.info(f"  Architecture: {'API' if has_existing_api == 'Yes' else 'Vector Search'}")
    logger.info(f"  Catalog size: {catalog_size}")
    logger.info(f"  Search type:  {search_type}")
    logger.info(f"  Images:       {has_images}")
    logger.info(f"  Voice:        {voice}")
    logger.info(f"  UI:           {ui_type}")

    ok = True

    # --- Path A: API Integration ---
    if has_existing_api in ("Yes", "Both"):
        if Path("scripts/api_connector.py").exists():
            ok = run_step(
                "Configuring API connector...",
                ["python", "scripts/api_connector.py", "--config", config_path, "--project-id", project_id],
                dry_run,
            ) and ok

    # --- Path B: Database + Vector Search ---
    if has_existing_api in ("No", "Both"):
        # Validate sample data
        sample_file = None
        for candidate in ["assets/sample-products.csv", "data/sample-products.csv", "data/products.csv"]:
            if Path(candidate).exists():
                sample_file = candidate
                break

        if sample_file:
            ok = run_step(
                "Validating sample data...",
                ["python", "scripts/validate_schema.py", "--file", sample_file, "--fields-level", product_fields],
                dry_run,
            ) and ok

            if ok:
                ok = run_step(
                    "Ingesting into BigQuery...",
                    ["python", "scripts/ingest_bigquery.py", "--config", config_path,
                     "--project-id", project_id, "--dataset-id", dataset_id,
                     "--table-id", table_id, "--local-file", sample_file],
                    dry_run,
                ) and ok

            if ok:
                ok = run_step(
                    "Creating Vector Search index...",
                    ["python", "scripts/ingest_vertex_search.py", "--config", config_path,
                     "--project-id", project_id],
                    dry_run,
                ) and ok

            # Large catalog warning
            if catalog_size == "500K+":
                logger.info("\n  NOTE: Catalog size 500K+ detected.")
                logger.info("  Consider using Dataflow for batch ingestion instead of direct inserts.")
                logger.info("  See references/architecture.md for large catalog patterns.")
        else:
            logger.info("\n  No sample data file found. Skipping ingestion.")

    # --- Image ingestion (if enabled) ---
    if has_images not in ("No", "No images"):
        if Path("scripts/ingest_gcs.py").exists():
            ok = run_step(
                "Uploading product images to GCS...",
                ["python", "scripts/ingest_gcs.py", "--config", config_path, "--project-id", project_id],
                dry_run,
            ) and ok

        # Multimodal search requires images
        if search_type == "Multimodal":
            logger.info("\n  Multimodal search enabled -- image embeddings will be generated alongside text.")

    # --- Voice search setup ---
    if voice == "Yes":
        if Path("scripts/live_search.py").exists():
            logger.info("\n  Voice search enabled. Test with:")
            logger.info(f"    python scripts/live_search.py --config {config_path} --project-id {project_id} --mode text")
            logger.info(f"    python scripts/live_search.py --config {config_path} --project-id {project_id} --mode voice")
        else:
            logger.info("\n  Voice search enabled but live_search.py not found.")
            logger.info("  Re-run setup with voice enabled or copy live_search.py manually.")

    # --- Agent behavior config ---
    logger.info(f"\n  Agent behavior:")
    logger.info(f"    Vague queries: {vague_query_handling}")
    logger.info(f"    UI delivery:   {ui_type}")
    if ui_type == "API only":
        logger.info("    NOTE: API-only mode -- no web frontend will be generated.")
    elif ui_type == "None":
        logger.info("    NOTE: No UI -- agent will be available via API/SDK only.")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    if ok:
        logger.info("SETUP COMPLETE")
    else:
        logger.info("SETUP COMPLETED WITH ERRORS (check output above)")
    logger.info("=" * 60)

    return ok


def main():
    parser = argparse.ArgumentParser(description="Run setup pipeline from design-spec.md")
    parser.add_argument("--config", required=True, help="Path to design-spec.md")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without running")

    args = parser.parse_args()
    ok = setup(args.config, args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
