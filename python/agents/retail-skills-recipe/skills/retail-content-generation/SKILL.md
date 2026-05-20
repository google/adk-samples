---
name: retail-content-generation
description: >-
  Generates product content using Gemini: descriptions, titles, SEO metadata,
  marketing copy, A/B variants, and translations. Uses product attributes
  from BigQuery catalog. Layers on top of retail-product-search.
deps:
  - source: retail-product-search
    type: skill
  - source: agents-cli
    type: cli
    url: https://github.com/google/agents-cli
    install: pip install google-agents-cli && agents-cli setup
priority: high
conflicts:
  - adk-scaffold
  - google-agents-cli-workflow
  - google-agents-cli-scaffold
  - google-agents-cli-adk-code
  - superpowers:brainstorming
  - superpowers:writing-plans
  - superpowers:executing-plans
  - superpowers:test-driven-development
trigger_phrases:
  - product description generation
  - product SEO copy
  - product marketing copy
  - product content generation
---

# Product Content Generation

Generates product content at scale using Gemini. Layers on top of
`retail-product-search` -- reads product data from BigQuery and generates
enriched content.

## ⚠️ Precedence (read first, supersedes other skills)

**Primary routing comes from `AGENTS.md` at the project root** (written by
the installer). The content below is a fallback for agents that don't
read AGENTS.md.

When the user wants to add content generation to a retail agent, **THIS
skill is the authoritative source for the conversation flow.** If
`adk-scaffold`, `google-agents-cli-workflow`, or any general agent-setup
skill is also active, do NOT use their bulk-question patterns. The first
action here is Q-MODE below.

## Execution Rules

1. **Q-MODE first, always.** Ask Q-MODE (below) before any other question.
2. **One question at a time.** Format every question as exactly
   `Q: <question>? [default: <value>]`. Accept empty input as "use the default."
   **NEVER ask multiple questions in one turn.**

## Setup Mode (Q-MODE — THIS IS THE FIRST QUESTION, ALWAYS)

**Self-confirmation (REQUIRED first output).** Your very first response MUST
be exactly the two lines below — no more, no less:

```
[skill: retail-content-generation] active. Ignoring any conflicting bulk-question flows from other skills.
Q-MODE: Pick a setup mode? [default: 1]
  1. Quick start  — 1 question, professional tone, default lengths, ~20s.
  2. Full setup   — full interview (brand voice, lengths, languages, A/B variants).
```

The user can say "configure more" or "customize" mid-flow to switch.

### Mode 1: Quick Start (1 question)

| Q | Question | Default |
|---|---|---|
| Q-A | Which content types to enable? | `descriptions, SEO titles, meta descriptions` |

Defaults taken silently:
- Base project path: `.`
- Brand voice: `professional`
- Description length: `100-150 words`
- SEO title: `< 60 chars, brand + key feature`
- Meta description: `< 160 chars, include CTA`
- Languages: `English only`
- A/B variants: `disabled`
- Batch size: `10`

After Q-A tell the user: "Say 'configure more' now to switch to Full setup
for marketing copy, translations, or A/B variants."

### Mode 2: Full Setup

Run the full interview in Steps 1-9 below.

Follow these rules strictly when executing this skill:

1. **Execute steps in order.** Do NOT jump ahead or skip steps.
2. **Verify each step succeeded** before moving to the next. If a command
   fails, stop and tell the user -- do NOT proceed.
3. **Wait for user answers.** Ask one question at a time. Do NOT assume
   defaults without confirming with the user.
4. **Confirm completion** of each step with the user before proceeding:
   "Step N is done. Ready for Step N+1?"

## When to Use

- Generating **product descriptions** from raw attributes (name, specs, category)
- Creating **SEO-optimized titles and meta descriptions** for product pages
- Writing **marketing copy** for ads, emails, or social media
- Producing **A/B test variants** of product descriptions
- **Translating** product content to multiple languages
- **Enriching** sparse catalog data with detailed descriptions

Prerequisite: `retail-product-search` must be set up first (products in BigQuery).

Do NOT use for generating product images (use retail-virtual-tryon),
pricing decisions, or inventory management.

## Content Types

1. **Product descriptions** -- detailed, feature-rich descriptions from attributes
2. **SEO titles** -- optimized product page titles (50-60 chars)
3. **Meta descriptions** -- search engine snippets (150-160 chars)
4. **Marketing copy** -- ad headlines, email subject lines, social captions
5. **A/B variants** -- multiple description versions for testing
6. **Translations** -- product content in target languages

## Step 1: Confirm Base Project

**C1.** Existing project path. Default: `.`

**C2.** Project name. Default: `product-search-agent`.

## Step 2: Content Types

**C3.** Which content types to generate? (select all that apply)
Default: Product descriptions + SEO titles + Meta descriptions.
Options: Product descriptions, SEO titles, Meta descriptions,
Marketing copy, A/B variants, Translations.

## Step 3: Brand Voice

**C4.** What is the brand's tone of voice?
Default: Professional and informative.
Options: Professional and informative, Casual and friendly,
Luxury and aspirational, Technical and detailed, Fun and playful.

**C5.** Brand name to use in content?
Default: (use brand field from product catalog).

**C6.** Any words or phrases to always include?
Default: None.
Example: "Free shipping", "Satisfaction guaranteed", "Handcrafted".

**C7.** Any words or phrases to never use?
Default: None.
Example: "Cheap", "Best in class", competitor brand names.

## Step 4: Description Settings

**C8.** Target description length?
Default: Medium (100-150 words).
Options: Short (50-75 words), Medium (100-150 words), Long (200-300 words).

**C9.** Include technical specifications in descriptions?
Default: Yes -- append specs as bullet points.

**C10.** Include use cases / "who it's for" section?
Default: Yes.

## Step 5: SEO Settings

Only if SEO titles or Meta descriptions selected in C3.

**C11.** Primary keyword strategy?
Default: Product category + brand + key feature.
Example: "Sony wireless noise-cancelling headphones".

**C12.** Include price in SEO title?
Default: No.

**C13.** Target locale for SEO?
Default: en-US.

## Step 6: Translation Settings

Only if Translations selected in C3.

**C14.** Target languages? (select all that apply)
Default: None.
Options: Spanish (es), French (fr), German (de), Japanese (ja),
Portuguese (pt-BR), Chinese Simplified (zh-CN), Korean (ko), Arabic (ar).

**C15.** Translation approach?
Default: Gemini direct translation (fast, good quality).
Options: Gemini direct translation, Gemini + human review workflow,
Cloud Translation API (cheaper, lower quality).

## Step 7: A/B Variant Settings

Only if A/B variants selected in C3.

**C16.** Number of variants per product?
Default: 2.
Options: 2, 3, 4.

**C17.** What should vary between versions?
Default: Tone and emphasis.
Options: Tone and emphasis, Length, Feature ordering, Call to action.

## Step 8: Processing Settings

**C18.** Gemini model for content generation?
Default: `gemini-2.5-flash` (fast, cost-effective).
Options: gemini-2.5-flash (fast), gemini-2.5-pro (higher quality).

**C19.** Batch size (products per generation run)?
Default: 50.

**C20.** Output format?
Default: BigQuery table (content_generated).
Options: BigQuery table, JSON file, CSV file.

## Step 9: GCP Configuration

**C21.** GCP project ID (must match product-search project).

**C22.** GCP region. Default: `us-central1`.

## Step 10: Add Files

The installer has merged the content-generation overlay files into the
existing `retail-product-search/` project:

```text
retail-product-search/
  app/
    content_agent.py            # NEW: ADK tool for content generation
    content_generator.py        # NEW: Gemini content generation logic
    agent.py                    # MODIFIED: generate_product_content wired into tools
  scripts/
    generate_content.py         # NEW: batch content generation
    export_content.py           # NEW: export generated content
```

If a file is missing (rare), fetch it from
`{{SOURCE_BASE}}/skills/retail-content-generation/<path>` as a fallback.

If the installer's auto-wire didn't take, edit `app/agent.py`:
1. Add `from app.content_agent import generate_product_content` to the imports
2. Append `generate_product_content` to the existing `tools=[…]` list

Add this tool to the existing `agent.py`:

```python
def generate_product_content(product_id: str, content_type: str = "description") -> str:
    """Generate content for a product using Gemini.

    Args:
        product_id: The product to generate content for.
        content_type: Type of content (description, seo_title, meta_description,
                      marketing_copy, translation).

    Returns:
        Generated content string.
    """
    from app.content_generator import generate
    return generate(product_id, content_type)
```

## Step 11: Test

Test queries:
1. `"Write a description for product prod-001"` -- basic generation
2. `"Create an SEO title for the wireless headphones"` -- SEO
3. `"Write 3 different descriptions for this product"` -- A/B variants
4. `"Translate the description for prod-001 to Spanish"` -- translation
5. `"Write a marketing headline for the laptop stand"` -- marketing copy

## Step 12: Evaluate

```bash
adk eval
```

Key metrics:
- Content quality (fluency, accuracy, brand consistency)
- SEO effectiveness (keyword inclusion, title length compliance)
- Factual accuracy (only uses attributes from catalog, no hallucination)
- Translation quality (natural, preserves meaning)

## Step 13: Batch Generation

Generate content for all products:

```bash
# Generate descriptions for all products
python scripts/generate_content.py --config design-spec.md --type description

# Generate SEO titles and meta descriptions
python scripts/generate_content.py --config design-spec.md --type seo

# Generate translations
python scripts/generate_content.py --config design-spec.md --type translation --languages es,fr,de

# Export to CSV
python scripts/export_content.py --config design-spec.md --format csv --output products_content.csv
```

## Gotchas

- **Hallucination**: Gemini may invent product features not in the catalog. ALWAYS validate against BigQuery attributes.
- **Brand consistency**: Test with 10-20 products first to calibrate tone before batch generation.
- **SEO length**: Titles > 60 chars get truncated in Google Search. Validate lengths.
- **Translation quality**: Machine translation may miss cultural nuances. Have native speakers review key products.
- **Cost**: gemini-2.5-pro costs ~10x more than flash. Use flash for batch, pro for hero products.
- **Rate limits**: Default 60 RPM. Add delays for large catalogs or request quota increase.

## References

- `references/architecture.md` (in retail-product-search sample) -- Base architecture
- `references/evaluation-guide.md` (in retail-product-search sample) -- Evaluation methodology
