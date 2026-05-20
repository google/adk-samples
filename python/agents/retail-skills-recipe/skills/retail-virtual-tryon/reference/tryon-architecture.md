# Virtual Try-On Architecture

## Overview

Try-on supports two model paths via the same `generate_tryon()` entry point:

```
User photo + Product image
    |
    v
tryon_processor.generate_tryon()
    |-- Fetch product image from GCS
    |-- Pre-flight classifier (product cutout vs person)  → adjusts safety
    |
    |-- IF model is virtual-try-on-001 (default for apparel):
    |     -- client.models.recontext_image(...)
    |     -- RecontextImageSource(person_image, product_images)
    |
    |-- ELSE (gemini-*-image, when text prompting needed):
    |     -- client.models.generate_content(...)
    |     -- multi-modal contents = [person, product, prompt]
    |     -- response_modalities=["TEXT", "IMAGE"]
    |
    |-- Upload all variations to GCS
    v
Agent shows result to user
```

## Key Components

### tryon_agent.py
- ADK tool the agent calls when user says "try this on"
- Reads model + safety level from environment variables (`GEMINI_IMAGE_MODEL`, `TRYON_SAFETY_LEVEL`)
- Returns try-on image URI for display

### tryon_processor.py
- Two-path generator: `_generate_with_vto` (Imagen) vs `_generate_with_gemini`
- `is_vto_model()` routes calls to the right API
- Pre-flight product-cutout classifier (reduces safety false positives, Gemini path only)
- Manages GCS input/output

## SDK: google-genai (NOT vertexai.generative_models)

Always use the new unified [`google-genai`](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/image/overview) Python SDK. Do not use `vertexai.generative_models` or `vertexai.preview.vision_models` — those are legacy.

```bash
pip install --upgrade google-genai
```

Set `GOOGLE_GENAI_USE_VERTEXAI=True` to route through Vertex AI (`tryon_processor._make_client` does this automatically).

## Path A: Dedicated VTO model — `virtual-try-on-001` (default)

Image-only, purpose-built, highest fidelity for apparel. Up to 4 images per request, 50 regional RPM quota. No text prompt input.

```python
import os
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")

response = client.models.recontext_image(
    model="virtual-try-on-001",
    source=types.RecontextImageSource(
        person_image=types.Image(image_bytes=person_bytes, mime_type="image/jpeg"),
        product_images=[
            types.ProductImage(
                product_image=types.Image(image_bytes=product_bytes, mime_type="image/jpeg")
            ),
        ],
    ),
    config=types.RecontextImageConfig(number_of_images=2),
)

for gen in response.generated_images:
    output_bytes = gen.image.image_bytes  # bytes ready to upload to GCS
```

## Path B: Gemini Image Generation

```python
import os
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")

# flash: fastest, cheapest, GA (gemini-2.5-flash-image)
# pro: best quality, GA (gemini-2.5-pro-image)
response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[
        types.Part.from_bytes(data=user_photo_bytes, mime_type="image/jpeg"),
        types.Part.from_bytes(data=product_image_bytes, mime_type="image/jpeg"),
        f"Show this person wearing the {product_description}. "
        "Keep face, body, and background the same. Photorealistic.",
    ],
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
    ),
)

# Extract generated image from response
for part in response.candidates[0].content.parts:
    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
        output_bytes = part.inline_data.data
```

## Pre-flight Product Classifier

Before calling the generation API, `tryon_processor.py` runs a lightweight
classifier with `gemini-2.5-flash` to detect product cutouts:

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        "Is this a product photo or a person photo? Answer: 'product' or 'person'.",
    ],
)
```

If it's a product cutout, the safety filter is relaxed one step (block_most → block_some)
to prevent false positives on white-background fashion images.

## Privacy Considerations

1. User photos are stored in a separate bucket with 24-hour auto-delete
2. Photos are never used for training or shared with third parties
3. Privacy notice is displayed before requesting photo upload
4. Users can opt-in to save photos for future sessions

## Latency Budget

| Step | Target |
|------|--------|
| Photo upload | < 1s |
| Pre-flight classifier | ~0.5s |
| Image generation (flash) | ~2s |
| Image generation (pro) | ~4-5s |
| Result delivery | < 0.5s |
| **Total (flash)** | **< 4s P90** |
| **Total (pro)** | **< 7s P90** |
