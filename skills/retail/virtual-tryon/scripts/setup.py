#!/usr/bin/env python3
"""Automated project setup for virtual try-on, driven by design-spec.md.

Reads the design-spec.md configuration and runs setup_tryon.py.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

try:
    from _setup_utils import load_config, run_step
except ImportError:
    # Minimal fallback loader if _setup_utils isn't importable
    def load_config(path):
        import re

        import yaml

        try:
            content = Path(path).read_text()
            parts = re.split(r"^---\s*$", content, flags=re.MULTILINE)
            if len(parts) >= 3:
                return yaml.safe_load(parts[1]) or {}
        except Exception:
            pass
        return {}

    def run_step(msg, cmd, dry_run=False):
        print(msg)
        if dry_run:
            print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
            return True
        res = subprocess.run(cmd, check=False)
        return res.returncode == 0


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def setup(config_path: str, dry_run: bool = False):
    """Run the virtual try-on setup pipeline based on design-spec.md."""
    cfg = load_config(config_path)

    project_id = cfg.get("gcp_project_id", "")
    if not project_id:
        logger.error("gcp_project_id is required in design-spec.md")
        sys.exit(1)

    gcp_region = cfg.get("gcp_region", "us-west1")
    tryon_categories = cfg.get("tryon_categories", ["Clothing"])
    tryon_mode = cfg.get("tryon_mode", "image_and_video")
    image_model = cfg.get("tryon_model", "flash")
    output_bucket = (
        cfg.get("tryon_output_bucket") or f"{project_id}-tryon-output"
    )
    upload_bucket = (
        cfg.get("tryon_upload_bucket") or f"{project_id}-tryon-uploads"
    )

    logger.info("=" * 60)
    logger.info("VIRTUAL TRY-ON SETUP (driven by design-spec.md)")
    logger.info("=" * 60)
    logger.info(f"  Project:            {project_id}")
    logger.info(f"  Region:             {gcp_region}")
    logger.info(f"  Categories:         {tryon_categories}")
    logger.info(f"  Try-On Mode:        {tryon_mode}")
    logger.info(f"  Image Model:        {image_model}")
    logger.info(f"  Output GCS Bucket:  {output_bucket}")
    logger.info(f"  Upload GCS Bucket:  {upload_bucket}")

    tryon_catalog_path = cfg.get("tryon_catalog_path", "catalog_images")
    logger.info(f"  Catalog Path:       {tryon_catalog_path}")

    setup_tryon_path = str(Path(__file__).resolve().parent / "setup_tryon.py")
    cmd = [
        "python",
        setup_tryon_path,
        "--config",
        config_path,
        "--project-id",
        project_id,
        "--location",
        gcp_region,
        "--output-bucket",
        output_bucket,
        "--upload-bucket",
        upload_bucket,
        "--model",
        image_model,
        "--catalog",
        tryon_catalog_path,
    ]

    catalog_upload = cfg.get("tryon_catalog_upload", True)
    if catalog_upload is False:
        cmd.append("--no-upload")

    if tryon_mode in ("image_and_video", "video"):
        cmd.append("--video")

    ok = run_step(
        "\nRunning GCP resource setup & API verification...",
        cmd,
        dry_run,
    )

    export_dir = cfg.get("export_directory")
    if ok and export_dir:
        export_app_path = str(Path(__file__).resolve().parent / "export_app.py")
        export_cmd = [
            "python",
            export_app_path,
            "--config",
            config_path,
            "--skill-dir",
            str(Path(__file__).resolve().parent.parent),
        ]
        ok = run_step(
            f"\nExporting standalone container application to: {export_dir}...",
            export_cmd,
            dry_run,
        )

    logger.info("\n" + "=" * 60)
    if ok:
        logger.info("VIRTUAL TRY-ON SETUP COMPLETED SUCCESSFULLY")
        logger.info("To start the agent run: adk web .")
        if export_dir:
            logger.info(f"Standalone application exported to: {export_dir}")
            logger.info(
                f"Navigate to '{export_dir}' and run 'deploy_cloudrun.sh' to deploy to Cloud Run."
            )
    else:
        logger.info("SETUP COMPLETED WITH ERRORS")
    logger.info("=" * 60)
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Run virtual try-on setup from design-spec.md"
    )
    parser.add_argument(
        "--config", required=True, help="Path to design-spec.md"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without running"
    )
    args = parser.parse_args()
    ok = setup(args.config, args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
