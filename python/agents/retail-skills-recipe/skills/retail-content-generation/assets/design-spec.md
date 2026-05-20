---
# Product Content Generation Agent - Design Spec
# Extends the product search design spec with content generation settings.

# --- Content Types ---
content_type: "description"  # description | seo_title | meta_description | marketing_copy | translation | all
content_types:
  - "Product descriptions"
  - "SEO titles"
  - "Meta descriptions"

# --- Brand Voice ---
brand_tone: "Professional and informative"
brand_name: ""  # Uses brand field from catalog if empty
always_include: ""  # e.g. "Free shipping, Satisfaction guaranteed"
never_use: ""  # e.g. "Cheap, Best in class"

# --- Description Settings ---
description_length: "Medium (100-150 words)"
include_specs: true
include_use_cases: true

# --- SEO Settings ---
include_price_in_seo: false
seo_locale: "en-US"

# --- Translation Settings ---
target_languages: []  # e.g. ["es", "fr", "de"]
translation_approach: "Gemini direct translation"

# --- A/B Variants ---
ab_variants: 2
variant_strategy: "Tone and emphasis"

# --- Processing ---
gemini_model: "gemini-2.5-flash"
batch_size: 50
output_format: "BigQuery table"

# --- GCP Configuration (must match product search) ---
gcp_project_id: ""  # REQUIRED: your GCP project ID
gcp_region: "us-central1"
---

# Product Content Generation Agent

## Overview

This design spec captures the configuration decisions for a product content
generation agent that layers on top of the product search agent.

## Prerequisites

- Product search agent must be set up first (see `skills/retail-product-search/`)
- Products must be ingested into BigQuery

## How to Use

1. Fill in the YAML frontmatter above
2. Pass this file to scripts: `python scripts/generate_content.py --config design-spec.md`
