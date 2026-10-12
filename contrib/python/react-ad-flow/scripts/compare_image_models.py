"""Side-by-side image-model comparison for Phase 14a.

Compares gemini-3-pro-image (current default) vs gemini-3.1-flash-lite-image
(Nano Banana 2 Lite) across the call shapes that share the IMAGE_GENERATION
knob: Stage-1 scene images (fashion wearable + retail-core non-wearable) and
the chart/infographic shape (IMAGE-only modalities, 16:9) used by
metrics_tools/maps_tools.

Run from the repo root with app/.env loaded:

    set -a; source app/.env; set +a
    .venv/bin/python scripts/compare_image_models.py

Outputs PNGs + metrics.json under generated/model-comparison/ (gitignored).
"""

import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai
from google.genai import types
from PIL import Image

from app.models.product import Product
from app.models.variation import CreativeVariation
from app.tools.prompt_builders import build_scene_image_prompt

MODELS = ["gemini-3-pro-image", "gemini-3.1-flash-lite-image"]
OUT_DIR = (
    Path(__file__).resolve().parent.parent / "generated" / "model-comparison"
)

FASHION_PRODUCT = {
    "name": "sage-satin-camisole",
    "details": "A sage green satin camisole with delicate lace trim",
    "category": "top",  # wearable archetype (seeded catalog category)
    "color": "sage green",
    "fabric": "satin",
}
RETAIL_PRODUCT = {
    "name": "aurora-cold-brew-330ml",
    "details": "A sleek 330ml can of Aurora cold brew coffee, matte black with aurora-gradient accents",
    "category": "beverage",
    "color": "matte black",
}
CHART_PROMPT = (
    "A clean 16:9 infographic for a retail dashboard titled 'RPI Across Creatives'. "
    "A horizontal bar chart with three bars labeled studio-minimalist (0.0605), "
    "golden-hour-rooftop (0.0595), urban-street (0.0589), axis label "
    "'Revenue per Impression (USD)', legible sans-serif text, light background."
)


def scene_case(product: dict) -> tuple[str, dict]:
    variation = CreativeVariation(name="model-comparison")
    # build_scene_image_prompt takes a typed Product (Phase 8); the row-dict
    # shape above converts via from_row (image_filename is a required column).
    typed = Product.from_row(
        {**product, "image_filename": f"{product['name']}.png"}
    )
    prompt = build_scene_image_prompt(typed, variation)
    return prompt, {"response_modalities": ["image", "text"]}


def chart_case() -> tuple[str, dict]:
    return CHART_PROMPT, {
        "response_modalities": ["IMAGE"],
        "image_config": types.ImageConfig(aspect_ratio="16:9"),
    }


CASES = {
    "fashion-scene": scene_case(FASHION_PRODUCT),
    "retail-scene": scene_case(RETAIL_PRODUCT),
    "chart-16x9": chart_case(),
}


def reference_case() -> tuple[list, dict] | None:
    """Fashion scene WITH a reference product image, mirroring
    generate_scene_image's wearable branch (video_tools.py:233-254). Returns
    None when no reference image is reachable (local-first with no bundle) —
    then both models are compared referenceless, symmetrically, and the doc
    must say so."""
    try:
        from app import storage

        filename = "sage-satin-camisole.png"
        if not storage.product_image_exists(filename):
            return None
        image_bytes = storage.read_product_image(filename)
    except Exception:
        return None
    prompt, cfg = scene_case(FASHION_PRODUCT)
    preamble = (
        "Use the provided image as the exact visual reference for the garment. "
        "The model must be wearing this exact garment.\n\n"
    )
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        preamble + prompt,
    ]
    return contents, cfg


_ref = reference_case()
if _ref is not None:
    CASES["fashion-scene-with-reference"] = _ref
else:
    print(
        "NOTE: no reference product image reachable — with-reference case skipped, "
        "comparison is referenceless for both models (record this in the doc)"
    )


def extract_image_bytes(response) -> bytes | None:
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            return part.inline_data.data
    return None


def main() -> int:
    client = genai.Client()
    metrics: dict[str, dict] = {}
    for model in MODELS:
        (OUT_DIR / model).mkdir(parents=True, exist_ok=True)
        for case, (prompt, cfg) in CASES.items():
            t0 = time.monotonic()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**cfg),
            )
            latency = round(time.monotonic() - t0, 1)
            image_bytes = extract_image_bytes(response)
            if image_bytes is None:
                metrics[f"{model}/{case}"] = {
                    "error": "no image returned",
                    "latency_s": latency,
                }
                print(f"FAIL  {model} {case}: no image ({latency}s)")
                continue
            out_path = OUT_DIR / model / f"{case}.png"
            out_path.write_bytes(image_bytes)
            with Image.open(io.BytesIO(image_bytes)) as im:
                resolution = f"{im.width}x{im.height}"
            metrics[f"{model}/{case}"] = {
                "bytes": len(image_bytes),
                "resolution": resolution,
                "latency_s": latency,
                "path": str(out_path),
            }
            print(
                f"OK    {model} {case}: {resolution}, {len(image_bytes)} bytes, {latency}s"
            )
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics written to {OUT_DIR / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
