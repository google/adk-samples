---
name: retail-product-recommendation
description: >-
  Adds product recommendation capabilities to an existing product search agent.
  Supports collaborative filtering, content-based, Vertex AI Recommendations AI,
  and LLM-driven recommendations. Use when adding "you might also like",
  "frequently bought together", or personalized homepage recommendations.
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
  - product recommendations
  - you might also like
  - frequently bought together
  - personalized homepage recommendations
---

# Product Recommendation

Adds recommendation capabilities to an existing product search agent.
Layers on top of `retail-product-search` -- only adds delta files.

## ⚠️ Precedence (read first, supersedes other skills)

**Primary routing comes from `AGENTS.md` at the project root** (written by
the installer). The content below is a fallback for agents that don't
read AGENTS.md.

When the user wants to add product recommendations to a retail agent,
**THIS skill is the authoritative source for the conversation flow.**
If `adk-scaffold`, `google-agents-cli-workflow`, or any general agent-setup
skill is also active, do NOT use their bulk-question patterns. The first
action here is Q-MODE below.

## When to Use

- Adding **"You might also like"** or **"Frequently bought together"**
- Implementing **personalized homepage** recommendations
- Using **Vertex AI Recommendations AI** for production-grade recs

Prerequisite: `retail-product-search` must be set up first.

Do NOT use for generic content recommendations, non-retail domains, or
standalone recommendations without an existing search agent.

## Recommendation Types

1. **Collaborative filtering** -- "Users who bought X also bought Y." Needs user_events table.
2. **Content-based** -- "Similar products by attributes." Uses existing product embeddings.
3. **Vertex AI Recommendations AI** -- Production-grade via Google Retail API.
4. **LLM-driven** -- Gemini suggests from conversation context + catalog.

## Execution Rules

1. **Q-MODE first, always.** Ask Q-MODE (below) before any other question.
2. **One question at a time.** Format every question as exactly
   `Q: <question>? [default: <value>]`. Accept empty input as "use the default."
   **NEVER ask multiple questions in one turn.**

## Setup Mode (Q-MODE — THIS IS THE FIRST QUESTION, ALWAYS)

**Self-confirmation (REQUIRED first output).** Your very first response MUST
be exactly the two lines below — no more, no less:

```
[skill: retail-product-recommendation] active. Ignoring any conflicting bulk-question flows from other skills.
Q-MODE: Pick a setup mode? [default: 1]
  1. Quick start  — 1 question, content-based + LLM-driven defaults, ~20s.
  2. Full setup   — full interview (strategy, top-N, context, Vertex AI, etc.).
```

The user can say "configure more" or "customize" mid-flow to switch.

### Mode 1: Quick Start (1 question)

| Q | Question | Default |
|---|---|---|
| Q-A | Have user-event data yet? | `No` (content-based + LLM-driven only) |

Defaults taken silently:
- Base project path: `.`
- Recommendation types: `content-based + LLM-driven`
- Recommendation context: `product page` ("you might also like")
- Top-N: `5`
- Vertex AI Recommendations AI: `off`

If user-event data arrives later, the user can re-run setup and switch to
collaborative filtering. After Q-A tell them: "Say 'configure more' now to
switch to Full setup with collaborative filtering or Vertex AI Rec AI."

### Mode 2: Full Setup

Run the full interview in Steps 1-7 below.

## Step 1: Confirm Base Project

**R1.** Existing project path. Default: `.`

**R2.** Project name. Default: `product-search-agent`.

## Step 2: Choose Recommendation Types

**R3.** Which types to enable? (select all that apply)
Default: collaborative + content-based + LLM-driven.
For Vertex AI Recommendations AI, see Step 6.

## Step 3: User Behavior Data

Only if collaborative filtering or Vertex AI Recommendations AI selected.

**R4.** User events source?
Default: No data yet (will start collecting).
Other options: BigQuery table, Pub/Sub, CSV/JSON export.

**R5.** Event types tracked?
Default: product views, add to cart, purchase.

## Step 4: Recommendation Context

**R6.** Recommendation placement -- where should recommendations appear?
Default: product detail page + cart page + homepage.

## Step 5: Settings

| # | Question | Default |
|---|----------|---------|
| R7 | Recommendations per context? | 5 |
| R8 | Personalization level? | Session-based |
| R9 | Cold start strategy? | Show popular/trending products |
| R10 | Respect search filters? | No -- show diverse options |

## Step 6: Vertex AI Recommendations AI (Conditional)

Only if selected in R3.

**R11.** Model types? Default: Others You May Like + Frequently Bought Together.

**R12.** Retail API catalog? Default: No -- will import from BigQuery.

Follow the [Vertex AI Recommendations AI documentation](https://cloud.google.com/retail/docs/recommendations-ai) for catalog setup.

## Step 7: GCP Configuration

**R13.** GCP project ID (must match product-search project).

**R14.** GCP region. Default: `us-central1`.

## Step 8: Add Files

The installer has merged the recommendation overlay files into the existing
`retail-product-search/` project:

```text
retail-product-search/
  app/
    recommendation_agent.py     # NEW
    recommendation_retriever.py # NEW
    agent.py                    # MODIFIED: recommend_products wired into tools
  scripts/
    ingest_user_events.py       # NEW (only used when collaborative filtering enabled)
```

If a file is missing (rare), fetch it from
`{{SOURCE_BASE}}/skills/retail-product-recommendation/<path>` as a fallback.

If the installer's auto-wire didn't take, edit `app/agent.py`:
1. Add `from app.recommendation_agent import recommend_products` to the imports
2. Append `recommend_products` to the existing `tools=[…]` list

Add this tool to the existing `agent.py`:

```python
def recommend_products(product_id: str, context: str = "") -> str:
    """Get product recommendations for a product or user context."""
    from app.recommendation_retriever import get_recommendations
    return get_recommendations(product_id, context)
```

## Step 9: Test

Test queries:
1. `"What else goes with this product?"` -- collaborative
2. `"Show me similar products"` -- content-based
3. `"What do you recommend?"` -- personalized homepage

## Step 10: Evaluate

```bash
adk eval
```

## Gotchas

- **Cold start**: Collaborative filtering needs minimum ~100 user events to produce useful results. Until then, fall back to content-based or trending.
- **Recommendation latency**: Target under 500ms P90.
- **Privacy**: Never expose other users' purchase data in recommendations.
- **Catalog only**: All recommended products must exist in the catalog.

## References

- `references/evaluation-guide.md` (in retail-product-search sample) -- Evaluation methodology
- `references/agent-example.md` (in retail-product-search sample) -- Base agent code
