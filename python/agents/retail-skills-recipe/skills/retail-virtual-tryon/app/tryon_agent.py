"""ADK tool for virtual try-on.

The agent calls try_on_product() when the user asks to try on a product.
Configure model and safety level via environment variables or design-spec.md.
"""

import os

from app.tryon_processor import generate_tryon


SUPPORTED_CATEGORIES = {"clothing", "eyewear", "jewelry", "cosmetics", "footwear"}


def try_on_product(product_id: str, user_photo_uri: str, product_description: str = "") -> dict:
    """Generate a virtual try-on image showing the product on the user.

    Args:
        product_id: The product to try on (must be in the catalog).
        user_photo_uri: GCS URI or base64 of the user's photo.
        product_description: Optional human-readable product name for the prompt.

    Returns:
        dict with keys: status ("success"/"error"), output_uri, model_used,
        variations, all_variations, error.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    output_bucket = os.getenv("TRYON_OUTPUT_BUCKET", "")
    upload_bucket = os.getenv("TRYON_UPLOAD_BUCKET", "")

    if not project_id or not output_bucket or not upload_bucket:
        return {
            "status": "error",
            "error": (
                "Try-on is not configured. Run `python scripts/setup_tryon.py` first "
                "and set GOOGLE_CLOUD_PROJECT, TRYON_OUTPUT_BUCKET, TRYON_UPLOAD_BUCKET."
            ),
        }

    try:
        result = generate_tryon(
            product_id=product_id,
            user_photo_uri=user_photo_uri,
            project_id=project_id,
            output_bucket=output_bucket,
            upload_bucket=upload_bucket,
            model_label_or_id=os.getenv("GEMINI_IMAGE_MODEL", "flash"),
            safety_level=os.getenv("TRYON_SAFETY_LEVEL", "block_most"),
            product_description=product_description,
        )
        return {
            "status": "success",
            "output_uri": result["output_uri"],
            "all_variations": result["all_variations"],
            "model_used": result["model_used"],
            "variations": result["variations_generated"],
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
