---
# Product Search Agent - Design Spec
# Fill in your values and use this as input for project generation.

# --- Data Model ---
industry: "General Retail"
project_name: "my-product-search"
product_fields: "Extended"  # Basic | Standard | Extended | Full
text_search_fields: "name, description"
filter_fields: "brand, category"
price_type: "Yes"  # Decimal prices
currency: "USD"
has_variants: "No"

# --- Search Architecture ---
# Set to "Yes" if you have an existing product search API,
# "No" to build vector search from scratch, or "Both".
has_existing_api: "No"

# --- Path B: Database + Vector Search (if has_existing_api is "No" or "Both") ---
database_type: "BigQuery"
catalog_size: "1K-50K"
catalog_change_frequency: "Daily (Cloud Scheduler cron)"
has_images: "No images"
search_type: "Text-only"
embedding_fields: "name, description, category, brand"
index_update_mode: "Batch update (scheduled rebuild, cheaper for large catalogs)"
embedding_model: "gemini-embedding-001"
collection_id: "retail-skill-products-collection"
gcp_region: "us-central1"
filter_strategy: "Pre-filter (Vertex AI metadata)"
delete_from_index: "Yes - hard delete via Vertex AI API (immediate removal)"
vector_db: "Vertex AI Vector Search (recommended)"

# --- Search Behavior ---
vague_query_handling: "Ask 1-2 clarifying questions before searching"
priority_filters: "Price range, Category, Brand"
default_sort: "Relevance (similarity score)"
show_out_of_stock: "No - apply Vertex AI metadata filter at query time"
multi_language: "No - single locale only"
remember_filters: "Yes - carry forward price ceiling, brand, in-stock preference"

# --- UI & Rendering ---
user_interface: "Cloud Run web app (React / HTML frontend)"
result_card_fields: "image, price, name, description, stock_badge, rating, add_to_cart"
results_per_page: "6"
brand_color: "#4285F4"
zero_results_action: "Vertex AI ANN fallback (widen similarity search radius)"

# --- Post-Search Actions ---
cart_integration: "Redirect to the product URL"
proactive_recommendations: "No - respond to explicit user searches only"
user_identity: "No - anonymous session, no personalisation"
voice_capabilities: "No"

# --- GCP Configuration ---
gcp_project_id: ""  # REQUIRED: your GCP project ID
billing_confirmed: "Yes - project billing enabled, API quotas set and reviewed"
---

# Product Search Agent

## Overview

This design spec captures the configuration decisions for a retail product search
agent built on Google Cloud. It is generated during the SKILL.md interview and
used by ingestion scripts and agent scaffolding.

## How to Use

1. Fill in the YAML frontmatter above (or let the coding agent do it conversationally)
2. Pass this file to any script: `python scripts/ingest_bigquery.py --config design-spec.md`
3. CLI args always override values from this file

## Design Decisions

Document any non-obvious choices here so future contributors understand the "why".
