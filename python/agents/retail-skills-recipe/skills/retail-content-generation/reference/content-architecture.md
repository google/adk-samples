# Product Content Generation Architecture

## Overview

Content generation reads product attributes from BigQuery and uses
Gemini to produce enriched content. The pipeline:

```
BigQuery (product attributes)
    |
    v
generate_content.py
    |-- Build prompt (product info + tone + constraints)
    |-- Call Gemini API
    |-- Validate output (length, no hallucination)
    |-- Store in BigQuery content_generated table
    v
export_content.py -> CSV/JSON for CMS import
```

## Content Generation Pipeline

### Single Product (Interactive)

```python
# In agent.py, user asks:
# "Write a description for the wireless headphones"

def generate_product_content(product_id, content_type):
    product = fetch_from_bigquery(product_id)
    prompt = build_prompt(product, content_type, brand_config)
    content = gemini.generate(prompt)
    return content
```

### Batch (All Products)

```bash
# Generate descriptions for entire catalog
python scripts/generate_content.py \
    --config design-spec.md \
    --type description \
    --batch-size 50

# Export for CMS import
python scripts/export_content.py \
    --config design-spec.md \
    --format csv \
    --output descriptions.csv
```

## Prompt Engineering

### Product Description Prompt

```
Write a {length} product description for:
Product: {name}
Category: {category}
Brand: {brand}
Price: ${price}
Description: {raw_description}

Tone: {brand_tone}
Always include: {always_include}
Never use: {never_use}

Include key features and benefits.
Include technical specifications as bullet points.
Include a "Who it's for" section.
```

### Quality Guardrails

1. **No hallucination**: Only use attributes from the catalog
2. **Length compliance**: SEO titles 50-60 chars, meta 150-160 chars
3. **Brand consistency**: Same tone across all products
4. **No competitor mentions**: Filter out competitor brand names

## Output Schema

### BigQuery: content_generated

| Field | Type | Description |
|-------|------|-------------|
| product_id | STRING | Product identifier |
| content_type | STRING | description, seo_title, meta_description, etc. |
| content | STRING | Generated content |
| language | STRING | Language code (en, es, fr, etc.) |
| model | STRING | Gemini model used |
| variant | INT64 | A/B variant number (1, 2, 3...) |
