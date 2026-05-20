"""Virtual try-on image generation, two paths:

1. **virtual-try-on-001** (Imagen) — purpose-built VTO model. Image-only inputs,
   highest fidelity for apparel. Default for clothing/footwear/standard items.
   API: client.models.recontext_image with RecontextImageSource

2. **gemini-{2.5-flash,2.5-pro}-image** -- prompt-based.
   Use when you need text-driven variations (color changes, scene, style).
   API: client.models.generate_content with multi-modal contents

Both use the new google-genai SDK. NEVER use vertexai.generative_models.
Set GOOGLE_GENAI_USE_VERTEXAI=True; the helpers below do this automatically.

Reference:
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/image/generate-virtual-try-on-images
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/imagen/virtual-try-on-001
"""

import logging
import os
import uuid

from google.cloud import storage

logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────

# Each entry: label -> {id, api}. `api` is the google-genai client method
# used to invoke the model: "recontext_image" (Imagen VTO) or "generate_content".
MODELS = {
    "vto":   {"id": "virtual-try-on-001",     "api": "recontext_image"},
    "flash": {"id": "gemini-2.5-flash-image", "api": "generate_content"},
    "pro":   {"id": "gemini-2.5-pro-image",   "api": "generate_content"},
}

# Safety levels (used by Gemini path; VTO has its own filter handling)
SAFETY_LEVELS = {
    "block_most": "BLOCK_LOW_AND_ABOVE",
    "block_some": "BLOCK_MEDIUM_AND_ABOVE",
    "block_few":  "BLOCK_ONLY_HIGH",
}

DEFAULT_LABEL = "vto"               # virtual-try-on-001 — best for clothing
DEFAULT_SAFETY_LEVEL = "block_most"
PRE_FLIGHT_MODEL = "gemini-2.5-flash"  # cheap classifier for product cutouts


def resolve_model(model_label_or_id: str) -> str:
    """Resolve a label (vto/flash/pro) or full model ID to a model ID."""
    if model_label_or_id in MODELS:
        return MODELS[model_label_or_id]["id"]
    return model_label_or_id or MODELS[DEFAULT_LABEL]["id"]


def is_vto_model(model_id: str) -> bool:
    """True if this model uses the recontext_image API (Imagen VTO)."""
    for entry in MODELS.values():
        if entry["id"] == model_id:
            return entry["api"] == "recontext_image"
    # Forward-compat: future virtual-try-on-* IDs also use recontext_image
    return model_id.startswith("virtual-try-on-")


def _make_client(project_id: str, location: str = "us-central1"):
    """Create a google-genai client routed through Vertex AI."""
    from google import genai
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
    return genai.Client(vertexai=True, project=project_id, location=location)


# ── Pre-flight classifier ────────────────────────────────────────────────────

def is_product_cutout(image_bytes: bytes, project_id: str) -> bool:
    """Detect product cutout (white/transparent background) vs. real person.

    Lets us selectively relax safety filters on cutouts to reduce false positives.
    """
    from google.genai import types

    client = _make_client(project_id)
    response = client.models.generate_content(
        model=PRE_FLIGHT_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            "Is this a product photo on a white or transparent background, "
            "or a photo of a real person wearing clothes? "
            "Answer with exactly one word: 'product' or 'person'.",
        ],
    )
    answer = (response.text or "").strip().lower()
    return answer.startswith("product")


# ── VTO via dedicated Imagen model (default) ─────────────────────────────────

def _generate_with_vto(
    client,
    person_bytes: bytes,
    product_bytes: bytes,
    num_variations: int,
):
    """Call virtual-try-on-001 via client.models.recontext_image."""
    from google.genai import types

    source = types.RecontextImageSource(
        person_image=types.Image(image_bytes=person_bytes, mime_type="image/jpeg"),
        product_images=[
            types.ProductImage(
                product_image=types.Image(image_bytes=product_bytes, mime_type="image/jpeg")
            ),
        ],
    )

    # virtual-try-on-001 supports up to 4 generated images per request
    config = types.RecontextImageConfig(number_of_images=min(num_variations, 4))

    response = client.models.recontext_image(
        model=MODELS["vto"]["id"],
        source=source,
        config=config,
    )

    images = []
    for gen in response.generated_images:
        if gen.image and gen.image.image_bytes:
            images.append(gen.image.image_bytes)
    return images


# ── VTO via Gemini image models (when text prompting needed) ─────────────────

def _generate_with_gemini(
    client,
    person_bytes: bytes,
    product_bytes: bytes,
    model_id: str,
    prompt: str,
    num_variations: int,
):
    """Call gemini-*-image via client.models.generate_content."""
    from google.genai import types

    images = []
    for _ in range(num_variations):
        response = client.models.generate_content(
            model=model_id,
            contents=[
                types.Part.from_bytes(data=person_bytes,  mime_type="image/jpeg"),
                types.Part.from_bytes(data=product_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                images.append(part.inline_data.data)
                break
    return images


# ── Top-level entry point ────────────────────────────────────────────────────

def generate_tryon(
    product_id: str,
    user_photo_uri: str,
    project_id: str,
    output_bucket: str,
    upload_bucket: str,
    model_label_or_id: str = "",
    safety_level: str = "",
    num_variations: int = 2,
    product_description: str = "",
) -> dict:
    """Generate a virtual try-on image.

    Routes to the dedicated VTO model by default (best for apparel) and falls
    back to Gemini image models when a label like 'flash'/'pro' is requested.

    Args:
        product_id: Product identifier (must exist in output_bucket as products/<id>.jpg).
        user_photo_uri: GCS URI (gs://...) or base64-encoded user photo.
        project_id: GCP project ID.
        output_bucket: GCS bucket for output images.
        upload_bucket: GCS bucket for ephemeral user uploads.
        model_label_or_id: 'vto' (default), 'flash', 'pro', or full model ID.
        safety_level: 'block_most' / 'block_some' / 'block_few' (Gemini path only).
        num_variations: Number of variations to generate (1–4).
        product_description: Used only for the Gemini path (text prompt).

    Returns:
        dict with output_uri, all_variations, model_used, safety_level, variations_generated.
    """
    model_id = resolve_model(
        model_label_or_id or os.getenv("GEMINI_IMAGE_MODEL", DEFAULT_LABEL)
    )
    effective_safety = safety_level or os.getenv("TRYON_SAFETY_LEVEL", DEFAULT_SAFETY_LEVEL)

    client = _make_client(project_id)
    gcs = storage.Client(project=project_id)

    # ── Fetch user photo ──────────────────────────────────────────────────────
    if user_photo_uri.startswith("gs://"):
        bucket_name, blob_path = user_photo_uri[5:].split("/", 1)
        photo_bytes = gcs.bucket(bucket_name).blob(blob_path).download_as_bytes()
    else:
        import base64
        photo_bytes = base64.b64decode(user_photo_uri)

    # ── Fetch product image ───────────────────────────────────────────────────
    product_blob = gcs.bucket(output_bucket).blob(f"products/{product_id}.jpg")
    product_bytes = product_blob.download_as_bytes()

    # ── Pre-flight classifier (informational; affects Gemini safety only) ─────
    product_is_cutout = False
    try:
        product_is_cutout = is_product_cutout(product_bytes, project_id)
    except Exception as e:
        logger.warning("Pre-flight classifier failed: %s. Proceeding without cutout detection.", e)

    adjusted_safety = effective_safety
    if product_is_cutout and effective_safety == "block_most" and not is_vto_model(model_id):
        adjusted_safety = "block_some"
        logger.info("Pre-flight: product is a cutout — relaxing safety to block_some for Gemini")

    # ── Route to the right API ────────────────────────────────────────────────
    output_image_bytes = []
    try:
        if is_vto_model(model_id):
            output_image_bytes = _generate_with_vto(
                client, photo_bytes, product_bytes, num_variations,
            )
        else:
            prompt = (
                f"Show this person wearing the {product_description or f'product {product_id}'}. "
                "Keep the person's face, body, and background exactly the same. "
                "Only change the clothing/accessory to the product shown. "
                "Photorealistic result."
            )
            output_image_bytes = _generate_with_gemini(
                client, photo_bytes, product_bytes, model_id, prompt, num_variations,
            )
    except Exception as e:
        safety_retry = os.getenv("TRYON_SAFETY_RETRY", "").lower() in ("true", "1", "yes")
        if ("SAFETY" in str(e).upper() and product_is_cutout
                and not is_vto_model(model_id) and safety_retry):
            logger.warning(
                "SAFETY RETRY: Safety block on cutout product (model=%s, "
                "product_id=%s, original_safety=%s). Retrying with block_few "
                "because TRYON_SAFETY_RETRY=true. Original error: %s",
                model_id, product_id, adjusted_safety, e,
            )
            adjusted_safety = "block_few"
            output_image_bytes = _generate_with_gemini(
                client, photo_bytes, product_bytes, model_id,
                f"Show person wearing {product_description or 'the product'}. Photorealistic.",
                num_variations,
            )
        else:
            raise

    if not output_image_bytes:
        raise RuntimeError(
            f"All {num_variations} try-on variations failed. "
            f"Model: {model_id}, safety: {adjusted_safety}. "
            f"See contrib/learnings/virtual-tryon-safety-filter.md."
        )

    # ── Upload all variations to GCS ──────────────────────────────────────────
    output_uris = []
    for img_bytes in output_image_bytes:
        blob_name = f"tryon-output/{product_id}/{uuid.uuid4()}.jpg"
        gcs.bucket(output_bucket).blob(blob_name).upload_from_string(
            img_bytes, content_type="image/jpeg",
        )
        output_uris.append(f"gs://{output_bucket}/{blob_name}")

    return {
        "output_uri": output_uris[0],
        "all_variations": output_uris,
        "model_used": model_id,
        "safety_level": adjusted_safety,
        "variations_generated": len(output_uris),
    }
