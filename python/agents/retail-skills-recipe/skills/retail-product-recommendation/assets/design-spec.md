---
# Product Recommendation Agent - Design Spec
# Extends the product search design spec with recommendation settings.

# --- GCP Configuration (must match product search) ---
gcp_project_id: ""  # REQUIRED: your GCP project ID
gcp_region: "us-central1"

# --- Recommendation Settings ---
recommendation_type: "collaborative"  # collaborative | content-based | hybrid | vertex-ai
has_user_events: "Yes"  # Yes | No (collaborative needs events, content-based doesn't)
rec_placement: "product-page"  # product-page | cart | homepage | all
max_recommendations: 10
user_events_table: "user_events"
dataset_id: "retail_dataset"
---

# Product Recommendation Agent

## Overview

This design spec captures the configuration decisions for a product recommendation
agent that layers on top of the product search agent.

## Prerequisites

- Product search agent must be set up first (see `skills/retail-product-search/`)
- User events data must be ingested

## How to Use

1. Fill in the YAML frontmatter above
2. Pass this file to scripts: `python scripts/ingest_user_events.py --config design-spec.md`
