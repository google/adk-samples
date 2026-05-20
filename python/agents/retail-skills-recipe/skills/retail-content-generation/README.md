# Retail Content Generation

Product content generation agent using Gemini for descriptions, SEO titles, marketing copy, and translations.

## Prerequisites

- Python 3.10+
- Google Cloud project with Vertex AI + BigQuery APIs enabled
- [retail-product-search](../retail-product-search/) set up first

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Generate product content
python scripts/generate_content.py --config assets/design-spec.md

# Export generated content
python scripts/export_content.py --config assets/design-spec.md --format csv
```

## Skill

See [skills/retail-content-generation/SKILL.md](../../skills/retail-content-generation/SKILL.md) for the conversational agent guide.

## License

Apache 2.0
