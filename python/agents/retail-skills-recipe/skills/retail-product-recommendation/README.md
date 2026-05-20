# Retail Product Recommendation

Product recommendation agent layered on top of product search. Supports collaborative filtering, content-based, and LLM-powered recommendations.

## Prerequisites

- Python 3.10+
- Google Cloud project with Vertex AI + BigQuery APIs enabled
- [retail-product-search](../retail-product-search/) set up first

## Setup

```bash
pip install -r requirements.txt
```

## Data Ingestion

```bash
# Ingest user behavior events
python scripts/ingest_user_events.py --config ../retail-product-search/assets/design-spec.md --local-file assets/sample-user-events.csv
```

## Skill

See [skills/retail-product-recommendation/SKILL.md](../../skills/retail-product-recommendation/SKILL.md) for the conversational agent guide.

## License

Apache 2.0
