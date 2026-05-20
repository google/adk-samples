---
# Virtual Try-On Agent - Design Spec
# Extends the product search design spec with try-on settings.

# --- Try-On Categories ---
tryon_categories:
  - "Clothing"
  - "Eyewear"

# --- Image Settings ---
has_transparent_images: "No"  # Yes | No — transparent PNGs give best overlay quality
product_image_source: "GCS bucket"
tryon_resolution: "512x512"   # 256x256 | 512x512 | 1024x1024
tryon_variations: 2           # 1 | 2 | 4

# --- User Photo Settings ---
user_photo_source: "Upload from device"   # Upload from device | Camera | Saved profile
store_user_photos: false                  # false = ephemeral (24-hour auto-delete)
privacy_notice: "Your photo is processed securely and not stored unless you opt in."

# --- Try-on Model ---
# Default: 'vto' = virtual-try-on-001 (dedicated Imagen VTO). Image-only, highest
# fidelity for apparel. Up to 4 variations per request. Uses recontext_image API.
#
# Switch to a Gemini tier ONLY when you need text prompt control (color changes,
# scene edits). Gemini tiers: flash / pro.
#
# Models:
#   vto        virtual-try-on-001              Dedicated VTO Imagen -- recommended
#   flash      gemini-2.5-flash-image          Prompt-based, fastest
#   pro        gemini-2.5-pro-image            Prompt-based, best quality
tryon_model: "vto"

# --- Guardrails ---
# block_most  — default, general retail (blocks medium and above)
# block_some  — fashion, activewear (blocks high only)
# block_few   — intimate apparel, adult fashion (blocks only explicit)
# Note: pre-flight product-cutout classifier always runs to reduce false positives.
safety_level: "block_most"

catalog_images_only: true     # Only allow try-on with products from the catalog

# --- GCP Configuration (must match product search project) ---
gcp_project_id: ""            # REQUIRED: your GCP project ID
gcp_region: "us-central1"
tryon_output_bucket: ""       # Default: {project_id}-tryon-output
tryon_upload_bucket: ""       # Default: {project_id}-tryon-uploads
---

# Virtual Try-On Agent

## Overview

This design spec captures configuration decisions for a virtual try-on agent
that layers on top of the product search agent.

## Prerequisites

- Product search agent must be set up first (`skills/retail-product-search/`)
- Product images available in GCS

## How to Use

1. Fill in `gcp_project_id` above
2. Choose `gemini_image_model` (flash is the default)
3. Choose `safety_level` based on your product categories
4. Run setup: `python scripts/setup_tryon.py --config assets/design-spec.md`

## Model Comparison

| Label | Model ID | API | Text prompt | Recommended for |
|-------|----------|-----|-------------|-----------------|
| `vto` ⭐ | `virtual-try-on-001` | `recontext_image` | No (image-only) | **Default** — clothing, footwear, apparel |
| `flash` | `gemini-2.5-flash-image` | `generate_content` | Yes | High-volume + text variations |
| `pro` | `gemini-2.5-pro-image` | `generate_content` | Yes | Editorial + complex scenes |

The dedicated VTO model is purpose-built and gives the highest fidelity for clothing. Reach for Gemini only when you need to drive variations with a text prompt (e.g. "show in red", "outdoor scene").

## Safety Level Guide

| Level | Config | Blocks | Use when |
|-------|--------|--------|----------|
| Block most | `block_most` | Medium + | General retail (default) |
| Block some | `block_some` | High only | Fashion, activewear |
| Block few | `block_few` | Explicit only | Intimate apparel |

The pre-flight classifier (`tryon_processor.py`) always runs first to distinguish
product cutouts from person photos — this significantly reduces false positives.
