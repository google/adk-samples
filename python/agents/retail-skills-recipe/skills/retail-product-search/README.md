# Retail Product Search

Semantic product search agent using Vertex AI Vector Search, BigQuery, and embeddings on Google Cloud.

## Prerequisites

- Python 3.10+
- Google Cloud project with Vertex AI + BigQuery APIs enabled
- `gcloud auth application-default login`

## Setup

```bash
pip install -r requirements.txt
```

## Data Ingestion

```bash
# Validate product data
python scripts/validate_schema.py --file assets/sample-products.csv --fields-level Extended

# Ingest into BigQuery
python scripts/ingest_bigquery.py --config assets/design-spec.md --local-file assets/sample-products.csv

# Create Vector Search index
python scripts/ingest_vertex_search.py --config assets/design-spec.md --bucket-name $PROJECT_ID-embeddings
```

## Using an Existing API

If you already have a product search API:

```bash
python scripts/api_connector.py --config assets/design-spec.md --api-url https://your-api.run.app --query "headphones"
```

## Cleanup

```bash
python scripts/cleanup.py --config assets/design-spec.md --dry-run    # preview
python scripts/cleanup.py --config assets/design-spec.md --confirm     # delete
```

## Running the Agent

Run all `adk` commands from this sample directory (`skills/retail-product-search/`).
The agent uses `from app.retrievers import ...`, which resolves only when the CWD
is this directory (or `skills/retail-product-search/` is on `PYTHONPATH`).

Text mode (default):
```bash
adk web .
```

Voice mode (uses Gemini Live native audio):
```bash
ENABLE_VOICE=true adk web .
```

You can also override the text model via `GEMINI_MODEL` env var.

## Skill

See [skills/retail-product-search/SKILL.md](../../skills/retail-product-search/SKILL.md) for the conversational agent guide.

## License

Apache 2.0
