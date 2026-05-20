#!/usr/bin/env python3
"""Automated project setup for virtual try-on, driven by design-spec.md.

Reads the design-spec.md configuration and runs the appropriate scripts
based on the user's answers.

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
    """Run the virtual try-on setup pipeline based on design-spec.md."""
    cfg = load_config(config_path)

    project_id = cfg.get("gcp_project_id", "")
    if not project_id:
        logger.error("gcp_project_id is required in design-spec.md")
        sys.exit(1)

    gcp_region = cfg.get("gcp_region", "us-central1")
    tryon_categories = cfg.get("tryon_categories", ["Clothing"])
    has_transparent = cfg.get("has_transparent_images", "No")
    user_photo_source = cfg.get("user_photo_source", "Upload from device")
    image_model = cfg.get("gemini_image_model", "flash")
    safety_level = cfg.get("safety_level", "block_most")
    tryon_resolution = cfg.get("tryon_resolution", "512x512")

    logger.info("=" * 60)
    logger.info("VIRTUAL TRY-ON SETUP (driven by design-spec.md)")
    logger.info("=" * 60)
    logger.info(f"  Project:            {project_id}")
    logger.info(f"  Region:             {gcp_region}")
    logger.info(f"  Categories:         {tryon_categories}")
    logger.info(f"  Transparent images: {has_transparent}")
    logger.info(f"  User photo mode:    {user_photo_source}")
    logger.info(f"  Image model:        {image_model}  (gemini-image)")
    logger.info(f"  Safety level:       {safety_level}")
    logger.info(f"  Resolution:         {tryon_resolution}")

    ok = True

    # Image preparation warnings
    if has_transparent == "No":
        logger.warning("\n  WARNING: Products do not have transparent background images.")
        logger.warning("  Virtual try-on works best with transparent PNG product images.")
        logger.warning("  Consider using a background removal tool before proceeding.")

    # Category-specific guidance
    for cat in tryon_categories:
        cat_lower = cat.lower()
        if cat_lower == "clothing":
            logger.info(f"\n  Category: {cat}")
            logger.info("  - Full-body product images recommended")
            logger.info("  - Front and back views improve results")
        elif cat_lower == "eyewear":
            logger.info(f"\n  Category: {cat}")
            logger.info("  - Front-facing product images required")
            logger.info("  - User photos should show face clearly")
        elif cat_lower == "jewelry":
            logger.info(f"\n  Category: {cat}")
            logger.info("  - High-resolution close-up images recommended")
            logger.info("  - Show product on neutral background")
        elif cat_lower == "cosmetics":
            logger.info(f"\n  Category: {cat}")
            logger.info("  - Product swatch images helpful")
            logger.info("  - User photos should have good lighting")

    # Run GCP resource setup
    ok = run_step(
        "Setting up GCP resources (buckets, Gemini image API)...",
        ["python", "scripts/setup_tryon.py", "--config", config_path, "--project-id", project_id],
        dry_run,
    ) and ok

    # User photo mode guidance
    if user_photo_source == "upload own":
        logger.info("\n  Users will upload their own photos.")
        logger.info("  Ensure privacy notice is displayed before photo upload.")
    else:
        logger.info("\n  Preset model photos will be used.")
        logger.info("  Prepare model photos for each supported category.")

    logger.info("\n" + "=" * 60)
    if ok:
        logger.info("VIRTUAL TRY-ON SETUP COMPLETE")
    else:
        logger.info("SETUP COMPLETED WITH ERRORS")
    logger.info("=" * 60)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Run virtual try-on setup from design-spec.md")
    parser.add_argument("--config", required=True, help="Path to design-spec.md")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without running")
    args = parser.parse_args()
    ok = setup(args.config, args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
