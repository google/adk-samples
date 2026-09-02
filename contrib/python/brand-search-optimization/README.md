# Brand Search Optimization

A multi-agent Agent Development Kit (ADK) recipe that optimizes retail product titles and analyzes brand visibility using **Gemini Computer Use**.

## Overview

Traditional web scraping struggles with dynamic search results, interactive filters, anti-bot protections, and modern sponsored layouts. This recipe leverages **Gemini Computer Use** (via Playwright browser automation) to inspect Search Engine Results Pages (SERPs) visually, audit organic vs. sponsored brand prominence, and generate data-driven product title enhancements.

## Architecture

The multi-agent workflow consists of:

```
                  ┌──────────────────────────────┐
                  │      Root Orchestrator       │
                  │ (brand_search_optimization)  │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│ Keyword Finding  │   │  Search Results  │   │      Comparison      │
│   (BigQuery)     │   │  (Computer Use)  │   │ (Generator & Critic) │
└──────────────────┘   └──────────────────┘   └──────────────────────┘
```

1. **`keyword_finding_agent`**: Queries product catalog data from BigQuery to identify top target keywords and brand attributes.
2. **`search_results_agent`**: Interacts with the browser environment via **Gemini Computer Use** to navigate to search engines, take visual screenshots, inspect SERP rankings, and extract live competitor product titles.
3. **`comparison_root_agent`**: Analyzes the gap between catalog listings and live top-ranking competitor titles, then produces optimized title suggestions through a generator/critic pattern.

## Prerequisites

- Python >= 3.11
- [`uv`](https://docs.astral.sh/uv/) package manager
- Google Cloud Project with Vertex AI enabled
- Application Default Credentials:

```bash
gcloud auth application-default login
```

## Setup

1. Navigate to the recipe directory:

```bash
cd contrib/python/brand-search-optimization
```

2. Copy the environment configuration:

```bash
cp .env.example .env
```

3. Install dependencies:

```bash
uv sync --dev
```

4. Install Playwright browser binaries (for Computer Use):

```bash
uv run playwright install chromium
```

## Running the Agent

### CLI Mode

```bash
uv run adk run brand_search_optimization
```

### Web UI Mode

```bash
uv run adk web
```

Then select `brand_search_optimization` from the UI menu.

## Validation & Testing

Run unit and runnability tests:

```bash
uv run pytest
```

Run code formatting and linting:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## Configuration

Key environment variables in `.env.example`:

- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID.
- `GOOGLE_CLOUD_LOCATION`: Google Cloud region (default: `us-central1`).
- `MODEL_NAME`: Target Gemini model (default: `gemini-3.7-flash`).
- `DATASET_ID` / `TABLE_ID`: BigQuery dataset and table for catalog products.