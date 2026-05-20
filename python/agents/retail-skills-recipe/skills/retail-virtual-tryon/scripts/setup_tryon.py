#!/usr/bin/env python3
"""Set up GCP resources for virtual try-on.

Creates GCS buckets for user photos and try-on output images,
verifies the Gemini image generation API, and confirms permissions.

Usage:
    python setup_tryon.py --config assets/design-spec.md
    python setup_tryon.py --project-id my-project --model flash
    python setup_tryon.py --project-id my-project --model pro --output-bucket my-tryon-output
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from google.cloud import storage

# Add _shared to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "_shared"))
from setup_utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dedicated VTO Imagen model — purpose-built, image-only, default for apparel
VTO_MODEL = "virtual-try-on-001"

# Gemini image generation tiers — prompt-based, use when text steering needed
GEMINI_IMAGE_MODELS = {
    "flash": "gemini-2.5-flash-image",
    "pro": "gemini-2.5-pro-image",
}

# All supported model labels combined ('vto' is the default)
ALL_MODELS = {
    "vto": VTO_MODEL,
    **GEMINI_IMAGE_MODELS,
}

# Safety level → Vertex AI HarmBlockThreshold (Gemini path only)
SAFETY_LEVELS = {
    "block_most": "BLOCK_LOW_AND_ABOVE",
    "block_some": "BLOCK_MEDIUM_AND_ABOVE",
    "block_few": "BLOCK_ONLY_HIGH",
}


def create_bucket_if_needed(project_id: str, bucket_name: str, location: str = "us-central1", auto_delete_days: int = 0) -> None:
    """Create a GCS bucket if it doesn't exist."""
    client = storage.Client(project=project_id)
    try:
        client.get_bucket(bucket_name)
        logger.info(f"Bucket {bucket_name} already exists")
    except Exception:
        logger.info(f"Creating bucket: {bucket_name}")
        bucket = client.create_bucket(bucket_name, location=location)
        if auto_delete_days > 0:
            bucket.add_lifecycle_delete_rule(age=auto_delete_days)
            bucket.patch()
            logger.info(f"Created bucket {bucket_name} with {auto_delete_days}-day auto-delete")
        else:
            logger.info(f"Created bucket {bucket_name}")


def resolve_model_id(model_label: str) -> str:
    """Resolve a label (vto/flash/pro) or a full model ID.

    Default: 'vto' → virtual-try-on-001 (the dedicated Imagen VTO model).
    """
    if model_label in ALL_MODELS:
        return ALL_MODELS[model_label]
    if model_label in ALL_MODELS.values() or model_label.startswith("virtual-try-on-"):
        return model_label
    logger.warning(f"Unknown model '{model_label}'. Using default: {VTO_MODEL}")
    return VTO_MODEL


def verify_gemini_image_api(project_id: str, model_id: str) -> bool:
    """Verify the Gemini image generation API is accessible.

    Uses the new google-genai SDK (NOT vertexai.generative_models). Sends a
    tiny text-only generate_content request to confirm the project has access
    to the chosen model.
    """
    try:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
        from google import genai

        client = genai.Client(vertexai=True, project=project_id, location="us-central1")
        # Lightweight verification — just resolve the model. We don't generate
        # to keep the cost zero; a 4xx here surfaces auth/access issues.
        client.models.get(model=model_id)
        logger.info(f"Gemini image model accessible: {model_id}")
        return True
    except Exception as e:
        logger.error(f"Gemini image API not accessible for model '{model_id}': {e}")
        logger.error("Enable Vertex AI at: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com")
        logger.error("Install the SDK: pip install --upgrade google-genai")
        return False


def print_model_comparison():
    """Print model tier comparison for reference."""
    logger.info("")
    logger.info("Try-on models:")
    logger.info("  vto        virtual-try-on-001                Default. Dedicated Imagen VTO. Image-only, highest")
    logger.info("                                                fidelity for apparel. Up to 4 variations / request.")
    logger.info("                                                Uses recontext_image API (no text prompt).")
    logger.info("")
    logger.info("Gemini image tiers (use when text-driven variations needed):")
    logger.info("  flash      gemini-2.5-flash-image            Fastest, 1290 tokens/img, good quality")
    logger.info("  pro        gemini-2.5-pro-image               Best quality, 2520 tokens/img")
    logger.info("")
    logger.info("Recommendation: use 'vto' for clothing/apparel, switch to a Gemini tier")
    logger.info("only when you need prompt-based control (color changes, scene edits).")
    logger.info("")


def setup(
    project_id: str,
    location: str,
    output_bucket: str,
    upload_bucket: str,
    model_id: str,
    safety_level: str,
) -> None:
    """Set up all GCP resources for virtual try-on."""
    logger.info(f"Setting up virtual try-on for project: {project_id}")
    logger.info(f"Image model: {model_id}")
    logger.info(f"Safety level: {safety_level}")

    # Output bucket for generated try-on images (no auto-delete)
    create_bucket_if_needed(project_id, output_bucket, location)

    # Upload bucket for user photos (24-hour auto-delete for privacy)
    create_bucket_if_needed(project_id, upload_bucket, location, auto_delete_days=1)

    # Verify API access
    if not verify_gemini_image_api(project_id, model_id):
        logger.warning("Gemini image API verification failed — try-on may not work")

    logger.info("")
    logger.info("Setup complete. Add these to your agent's environment:")
    logger.info(f"  TRYON_OUTPUT_BUCKET={output_bucket}")
    logger.info(f"  TRYON_UPLOAD_BUCKET={upload_bucket}")
    logger.info(f"  GEMINI_IMAGE_MODEL={model_id}")
    logger.info(f"  TRYON_SAFETY_LEVEL={safety_level}")


def main():
    parser = argparse.ArgumentParser(description="Set up GCP resources for virtual try-on")
    parser.add_argument("--config", default="", help="Path to design-spec.md")
    parser.add_argument("--project-id", help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="GCP region")
    parser.add_argument("--output-bucket", help="GCS bucket for try-on output images")
    parser.add_argument("--upload-bucket", help="GCS bucket for user photo uploads")
    parser.add_argument(
        "--model",
        default="",
        help="Try-on model: vto (default, virtual-try-on-001) | flash | pro",
        dest="model",
    )
    parser.add_argument(
        "--safety-level",
        default="",
        choices=list(SAFETY_LEVELS.keys()) + [""],
        help="Safety filter: block_most | block_some | block_few",
    )
    parser.add_argument("--list-models", action="store_true", help="Show available model tiers and exit")
    args = parser.parse_args()

    if args.list_models:
        print_model_comparison()
        sys.exit(0)

    # Load from config file first, CLI args override
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    if not args.project_id:
        args.project_id = cfg.get("gcp_project_id", "")
    if args.location == "us-central1" and cfg.get("gcp_region"):
        args.location = cfg["gcp_region"]
    if not args.output_bucket and cfg.get("tryon_output_bucket"):
        args.output_bucket = cfg["tryon_output_bucket"]
    if not args.upload_bucket and cfg.get("tryon_upload_bucket"):
        args.upload_bucket = cfg["tryon_upload_bucket"]

    # Model: CLI flag > design-spec > default vto
    # Reads either 'tryon_model' (preferred) or legacy 'gemini_image_model'
    model_label = args.model or cfg.get("tryon_model") or cfg.get("gemini_image_model") or "vto"
    model_id = resolve_model_id(model_label)

    # Safety level: CLI flag > design-spec > default block_most
    safety_level = args.safety_level or cfg.get("safety_level", "block_most")
    if safety_level not in SAFETY_LEVELS:
        logger.warning(f"Unknown safety level '{safety_level}'. Using block_most.")
        safety_level = "block_most"

    # Log try-on config
    tryon_cats = cfg.get("tryon_categories", [])
    has_transparent = cfg.get("has_transparent_images", "No")
    user_photo = cfg.get("user_photo_source", "Upload from device")
    if tryon_cats:
        logger.info(f"Try-on categories: {tryon_cats}")
    logger.info(f"Transparent images: {has_transparent}")
    logger.info(f"User photo mode: {user_photo}")
    if has_transparent == "No":
        logger.warning("Products lack transparent background images — try-on quality may be reduced.")

    if not args.project_id:
        parser.error("--project-id is required (or set gcp_project_id in design-spec.md)")

    if not args.output_bucket:
        args.output_bucket = f"{args.project_id}-tryon-output"
    if not args.upload_bucket:
        args.upload_bucket = f"{args.project_id}-tryon-uploads"

    setup(args.project_id, args.location, args.output_bucket, args.upload_bucket, model_id, safety_level)


if __name__ == "__main__":
    main()
