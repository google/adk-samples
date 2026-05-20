
---
name: retail-skills
description: >-
  Discovers and composes retail and e-commerce AI agent skills on Google Cloud.
  Use when building product search, recommendation, virtual try-on, or content
  generation agents. Covers Vertex AI, BigQuery, Vector Search, and Gemini
  (text + image generation) workflows.
---

# Retail Skills

Vertical skills for retail and e-commerce AI agents on Google Cloud.

## Available Skills

| Skill | Description | Depends On |
|-------|-------------|------------|
| [retail-product-search](retail-product-search/SKILL.md) | Semantic product search with Vector Search, RAG, optional voice | -- |
| [retail-product-recommendation](retail-product-recommendation/SKILL.md) | Product recommendations (collaborative, content-based, Vertex AI, LLM) | product-search |
| [retail-virtual-tryon](retail-virtual-tryon/SKILL.md) | Virtual try-on using Gemini image models — flash/pro tiers (clothing, eyewear, jewelry) | product-search |
| [retail-content-generation](retail-content-generation/SKILL.md) | Product descriptions, SEO, marketing copy, translations via Gemini | product-search |

## Getting Started

1. Start with **retail-product-search** to build a search agent
2. Layer on any combination of the other skills:
   - **retail-product-recommendation** for "you might also like"
   - **retail-virtual-tryon** for "try it on" with user photos
   - **retail-content-generation** for product descriptions and SEO at scale

## Skill Dependency Graph

```
retail-product-search (base)
    |
    +-- retail-product-recommendation
    |
    +-- retail-virtual-tryon
    |
    +-- retail-content-generation
```

All skills layer on top of product-search and can be combined independently.

## Adding New Skills

Each skill is a self-contained folder under `skills/`:

```
skills/{your-skill}/
    SKILL.md          # skill specification (the entry point)
    EVAL.yaml         # eval cases (rubric + assertions)
    reference/        # reference docs, guides
    scripts/          # setup automation, ingestion scripts
    assets/           # design-spec, sample data
    app/              # agent code (if applicable)
    tests/            # unit and integration tests
    pyproject.toml    # Python package config
    README.md         # skill-level documentation
```

1. Create `skills/{your-skill}/SKILL.md` with frontmatter
2. Add `scripts/`, `assets/`, `tests/` in the same directory
3. Create `EVAL.yaml` with at least 5 eval cases
4. Update [REGISTRY.md](REGISTRY.md) and the table above
