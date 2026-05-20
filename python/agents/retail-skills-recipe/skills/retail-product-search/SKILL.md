---
name: retail-product-search
description: >-
  Creates product search agents with semantic search, RAG, and voice on Google
  Cloud (Vertex AI Vector Search, BigQuery, embeddings). Use when building
  e-commerce search, catalog discovery, or shopping assistant agents. Handles
  data ingestion, multi-agent orchestration, evaluation, and deployment.
deps:
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
  - retail product search
  - e-commerce search agent
  - shopping search agent
  - product catalog search
---

# Product Search Agent

Creates product search agents with semantic search, RAG, and optional voice
on Google Cloud.

## ⚠️ Precedence (read first, supersedes other skills)

**Primary routing comes from `AGENTS.md` at the project root** (written by
the installer). It tells the agent that this skill drives the question flow
and that `google-agents-cli-*` skills handle mechanics underneath. The
content below is a fallback for agents that don't read AGENTS.md.

When the user wants a retail / e-commerce / product-catalog / shopping
search agent, **THIS skill is the authoritative source for the entire
conversation flow.**

If other skills are active in this session — for example `adk-scaffold`,
`google-agents-cli-workflow`, or any general "agent setup" skill — they
provide reference material for ADK and `agents-cli` mechanics, but they
**do NOT drive the question flow for retail use cases.** Specifically:

- **Do NOT use any other skill's question lists, "Step 1 / Step 2"
  groupings, or bulk question batches.** Use ONLY the questions defined
  in this SKILL.md.
- **The first action in this conversation is Q-MODE below.** Do not skip
  it. Do not merge it with other questions. Do not assume the user wants
  Full setup.
- **For installing `agents-cli`,** use only the command in the deps
  frontmatter: `pip install google-agents-cli && agents-cli setup`. Do
  NOT use `uvx` or any alternative form suggested by another skill.

## STOP — READ THIS BEFORE RESPONDING

**Your VERY FIRST response MUST be the Q-MODE block below. Nothing else.**

Do NOT ask about products, industry, GCP project, or anything else first.
Do NOT propose a plan. Do NOT explain what you will do. Do NOT say "Let me
help you build...". Do NOT jump to Step 1. Do NOT ask Q1.

Your first message to the user must be EXACTLY this (copy-paste, no changes):

```
[skill: retail-product-search] active.
Q-MODE: Pick a setup mode? [default: 1]
  1. Quick start -- 2 questions, smart defaults, ~60s. Best for demos and first-timers.
  2. Full setup  -- full interview (~10 questions, ~5 min). Best for real builds.
```

Then STOP and wait for the user's answer. Do not add anything after this block.

Accept: `1`, `quick`, empty/Enter (= Quick Start), `2`, or `full` (= Full Setup).

If you are about to output ANYTHING other than the block above as your first
response, you are violating this skill. Stop, delete your draft, and output
the block above instead.

## Execution Rules

Follow these rules strictly when executing this skill:

1. **Q-MODE first, always.** Your first response is the Q-MODE block above.
   No exceptions. No preamble. No plan proposals.
2. **One question at a time. Show the default. Accept empty input.**
   Format every question as exactly:
   ```
   Q: <question text>? [default: <value>]
   ```
   Pressing Enter with no input MUST be interpreted as "use the default."
   **NEVER ask multiple questions in one turn.** Do not number bulk lists
   ("1. ... 2. ... 3. ..."). Send one question, wait for the answer, then
   send the next.
3. **Execute steps in order.** Do NOT jump ahead or skip steps.
4. **Verify each step succeeded** before moving to the next. If a command
   fails, stop and tell the user -- do NOT proceed.
5. **Do NOT use `agents-cli data-ingestion`.** Use the retail skill scripts
   in `scripts/` instead.
6. **The retail skill files are already on disk** at `retail-product-search/`
   (the installer placed them there). Do NOT re-fetch or re-scaffold a
   sibling project -- use `agents-cli scaffold enhance .` in place (see
   Step 7).
7. **Save all answers to `assets/design-spec.md`** as you collect them.
   After the interview, run `python scripts/setup.py --config assets/design-spec.md`
   to execute the full pipeline. The setup script reads the design-spec and
   runs only the steps that match the user's choices.
8. **Confirm completion** of each step with the user before proceeding:
   "Step N is done. Ready for Step N+1?"

**At any point during Quick Start, the user can say "configure more" or
"customize" to switch to Full setup.** Carry over answers already given.

### Mode 1: Quick Start (2 questions)

Ask only these two; take defaults for everything else. Should take under 60 s.

| Q | Question | Default | Source of default |
|---|---|---|---|
| Q-A | GCP project ID? | value of `$GOOGLE_CLOUD_PROJECT`, else `gcloud config get-value project` | env / gcloud |
| Q-B | Where's your product data? | `assets/sample-products.csv` (bundled — 5 sample products) | bundled |

Accepted forms for Q-B:
- `<empty>` or `default` → use the bundled sample CSV
- `/path/to/file.csv` → local CSV file
- `gs://bucket/path/products.csv` → GCS object
- `bq://project.dataset.table` → existing BigQuery table (skip ingestion)

Defaults taken silently:
- Industry/project name: `retail-search`
- Product fields level: `Standard`
- Search architecture: Path B (Database + Vector Search)
- Catalog size: `1K-50K`
- Search type: `Text-only`
- Images: `No images`
- Voice: `No`
- UI: `Cloud Run web app`
- GCP region: `us-central1`
- Vague queries: `Ask 1-2 clarifying questions`

After Q-A and Q-B, tell the user: "Taking defaults for the rest. You can
change any of these later by editing `design-spec.md` and re-running setup,
or say 'configure more' now to switch to Full setup."

**Quick Start skips Steps 1-6 (interview) and Step 7 customization entirely.**
The sample `app/agent.py` and `app/retrievers.py` work out of the box -- do
NOT rewrite or modify them. Go directly to Step 8 (data ingestion) and then
Step 9 (test with `adk web .`).

### Mode 2: Full Setup

Run the full interview in Steps 1-6 below. Every question still shows a
`[default: ...]` and accepts empty input as "use the default." The user
just sees more questions, each individually skippable.

## When to Use

- Building **e-commerce product search** or **shopping assistant** agents
- Implementing **semantic catalog discovery** with natural language queries
- Adding **voice-based product search** (Gemini Live)

Do NOT use for generic document search (use ADK `agentic_rag`), simple keyword
search, or non-retail domains.

## Step 1: Understand the Product Domain

Ask these one at a time. Wait for answers before proceeding.

**Q1. Industry and project name**
Ask what products are searchable and what industry. Default: General Retail.
Project name defaults to `{industry}-search`.

**Q2. Product fields level**
Default: Standard (product_id, name, price, description, category, brand, image_url).
Extended adds rating, stock, manufacturer. Full adds variants, tags, specs, reviews.

**Q3. Searchable vs. filterable fields**
Free-text search fields (default: `name, description`) vs exact-match filters
(default: `brand, category`).

**Q4. Price config**
Default: decimal prices, USD.

**Q5. Variants**
Default: No.

## Step 2: Gate Question -- Search Architecture

Ask: "Do you have an existing product search API on GCP, or build from scratch?"

- **Yes** -> Path A (API integration)
- **No** (default) -> Path B (database + vector search)
- **Both** -> Path B then Path A

### Path A: API Integration

Only if user has an existing API. Ask progressively.

| # | Question | Default |
|---|----------|---------|
| A1 | API base URL? | (required) |
| A2 | Auth method? | Google OAuth |
| A3 | Search endpoint + HTTP method? | `GET /products/search` |
| A4 | Query parameters? | `q, category, brand, price_min, price_max, sort, page, limit` |
| A5 | Pagination? | `page + limit` |
| A6 | Supports vector search natively? | No |
| A7 | Zero results format? | Empty array with `total: 0` |
| A8 | Error handling? | Retry once with backoff |
| A9 | Rate limit? | Unknown |
| A10 | Cloud Logging? | Yes |
| A11 | Excluded categories? | None |

### Path B: Database + Vector Search

Only if building from scratch. Ask progressively.

| # | Question | Default |
|---|----------|---------|
| B1 | GCP database? | BigQuery |
| B2 | Catalog size? | 1K-50K. 500K+ may need Dataflow. |
| B3 | Catalog change frequency? | Daily (Cloud Scheduler) |
| B4 | Product images? | No images. If yes: ask about visual search + images per product. |
| B5 | Text-only or multimodal? | Text only |
| B6 | Fields to embed? | `name, description, category, brand` |
| B7 | Index update mode? | Batch (scheduled rebuild) |
| B8 | Embedding model? | `gemini-embedding-001` (auto-embedded by VS 2.0). |
| B9 | GCP region? | `us-central1` |
| B10 | Filter strategy? | Pre-filter (Vertex AI metadata) |
| B11 | Deletion handling? | Hard delete from index |
| B12 | Vector DB? | Vertex AI Vector Search |

## Step 3: Search Behavior

| # | Question | Default |
|---|----------|---------|
| S1 | Vague query handling? | Ask 1-2 clarifying questions |
| S2 | Priority filters for clarification? | Price range, Category, Brand |
| S3 | Sort order? | Relevance (similarity score) |
| S4 | Show out-of-stock? | No -- filter via Vertex AI metadata |
| S5 | Multi-language? | No |
| S6 | Remember filters across turns? | Yes |

## Step 4: UI and Rendering

| # | Question | Default |
|---|----------|---------|
| U1 | UI delivery? | Cloud Run web app |
| U2 | Result card fields? | image, price, name, description, stock badge, rating, add-to-cart |
| U3 | Results per page? | 6 |
| U4 | Brand color? | `#4285F4` |
| U5 | Zero results fallback? | Vertex AI ANN fallback (widen search radius) |

## Step 5: Post-Search Actions

| # | Question | Default |
|---|----------|---------|
| P1 | Add to Cart action? | Redirect to product URL |
| P2 | Proactive recommendations? | No |
| P3 | Logged-in user? | No -- anonymous session |
| P4 | Voice capabilities? | No |

## Step 6: GCP Configuration

| # | Question | Default |
|---|----------|---------|
| G1 | GCP project ID? | (required) |
| G2 | Billing confirmed? | Yes |
| G3 | GCP region? (if not set in Path B) | `us-central1` |

## Step 7: Scaffold the Project

The installer has already placed the retail skill files on disk at
`retail-product-search/` (relative to the user's working directory). Use
`agents-cli scaffold enhance` to add deployment + CI/CD scaffolding to that
existing tree **in place** -- do NOT run `agents-cli scaffold <name>`,
which creates a sibling directory and produces a duplicate tree.

```bash
cd retail-product-search
agents-cli scaffold enhance . \
  --agent-type agentic_rag \
  --datastore vertex_ai_vector_search \
  --deployment-target cloud_run \
  --prototype \
  -y
```

Deployment target mapping: CLI -> `none`, Web UI -> `cloud_run`,
Voice -> `agent_engine`, Custom API -> `cloud_run`.

**DO NOT run `agents-cli data-ingestion`.** It creates a generic document
pipeline that is incompatible with product catalogs. Use the retail skill
scripts under `scripts/` instead.

The project tree on disk should already look like this. If a file is
missing (e.g. user installed with `--minimal` in some future flag), fetch
it from `{{SOURCE_BASE}}/skills/retail-product-search/<path>` as a fallback.

```text
retail-product-search/         # current directory
  app/
    agent.py                   # Reference agent (rewrite as needed)
    retrievers.py              # Vector Search retrieval
  assets/
    design-spec.md             # Source of truth — fill in user's answers
    sample-products.csv        # Bundled demo catalog (5 products)
  references/                  # Deep-dive docs the agent can read on demand
  scripts/
    setup.py                   # Pipeline driver (reads design-spec.md)
    validate_schema.py
    ingest_bigquery.py
    ingest_vertex_search.py
    ingest_gcs.py              # multimodal only
    api_connector.py           # Path A only
    cleanup.py
    live_search.py             # voice only
    pubsub_sync.py             # real-time sync only
../_shared/                    # imported by scripts/setup.py
  setup_utils.py
```

Then customize the agent code:
1. Rewrite `app/agent.py` -- see [agent-example.md](references/agent-example.md)
2. Rewrite `app/retrievers.py` with product-specific fields
3. If Path A (existing API): configure `scripts/api_connector.py` with your API details
4. Add Pub/Sub sync if real-time -- see `scripts/pubsub_sync.py`
5. Add audio layer if voice -- see [audio-integration.md](references/audio-integration.md)

After copying, the project structure MUST look like this:

```text
{project_name}/
  app/
    agent.py              # Customized search agent
    retrievers.py         # Vector Search retrieval
    audio/                # (conditional: voice)
  assets/
    design-spec.md        # REQUIRED: source of truth
    sample-products.csv   # Bundled demo catalog (5 products)
  scripts/
    ingest_bigquery.py    # REQUIRED: from retail skill scripts/
    ingest_gcs.py         # From retail skill (multimodal only)
    ingest_vertex_search.py  # REQUIRED: from retail skill scripts/
    validate_schema.py    # REQUIRED: from retail skill scripts/
    api_connector.py      # From retail skill (Path A only)
    cleanup.py            # From retail skill scripts/
  # evalset lives at repo root: skills/retail-product-search/EVAL.yaml
  deployment/terraform/dev/
    products.tf
```

## Step 8: Configure and Ingest Data

**⚠️ DO NOT use `agents-cli data-ingestion`. It runs a generic document
pipeline that is INCOMPATIBLE with product catalogs. Use the retail skill
scripts below -- they are the ONLY correct ingestion method.**

First, set the GCP project ID in `assets/design-spec.md`:

```bash
cd {project_name}

# Set your GCP project ID (REQUIRED before running any script)
# Edit assets/design-spec.md and replace the empty gcp_project_id with your project
# Example: gcp_project_id: "my-gcp-project-123"
```

Then run these commands in order. Do NOT skip any step:

```bash
# Step 8a: Validate sample data (MUST pass before ingestion)
python scripts/validate_schema.py --file assets/sample-products.csv --fields-level Extended

# Step 8b: Load products into BigQuery
python scripts/ingest_bigquery.py --config assets/design-spec.md --local-file assets/sample-products.csv

# Step 8c: Create Vector Search 2.0 Collection and ingest products (2-5 min)
# This creates a Collection with auto-embeddings and inserts all products.
# No manual embedding generation needed -- VS 2.0 handles it automatically.
python scripts/ingest_vertex_search.py --config assets/design-spec.md
```

After Step 8c completes, set the `VECTOR_SEARCH_COLLECTION` environment
variable in your agent to the collection path printed by the script:

```bash
export VECTOR_SEARCH_COLLECTION="projects/$PROJECT_ID/locations/us-central1/collections/retail-skill-products-collection"
```

## Step 9: Test

```bash
adk web .
```

Test queries:
1. `"wireless headphones under $100"` -- price filter
2. `"I need a gift"` -- clarification flow
3. `"Which one has the best battery life?"` -- RAG detail
4. `"laptop for video editing"` -- semantic search

## Step 10: Evaluate

```bash
adk eval
```

Expect 5-10+ iterations. See [evaluation-guide.md](references/evaluation-guide.md)
for metrics and methodology.

## Step 11: Deploy

```bash
agents-cli scaffold enhance . \
  --deployment-target agent_engine \
  --cicd-runner github_actions \
  -y

make deploy
```

**Never deploy without explicit human approval.**

## Gotchas

- **Vector Search returns no results**: Collection empty or `VECTOR_SEARCH_COLLECTION`
  env var not set. Verify products were ingested by re-running `ingest_vertex_search.py`.
- **BigQuery ingestion fails**: Schema mismatch or missing required fields
  (product_id, name, price). Validate with `scripts/validate_schema.py`.
- **Slow search**: Use `tree_ah` algorithm for >1K products. Increase shard size.
- **Prototype first**: Use `--prototype` flag, add deployment infra later.
- **Real data only**: Don't test with placeholder/lorem ipsum products.
- **Prices are sacred**: Never fabricate or estimate prices.

## Completion Checklist

- [ ] Product domain understood (industry, fields, data source)
- [ ] Search architecture decided (API vs. vector search)
- [ ] Project scaffolded with agents-cli
- [ ] Data ingestion running, Vector Search populated
- [ ] Search + clarification agents working
- [ ] Evaluation passes success criteria
- [ ] Edge cases handled (see [architecture.md](references/architecture.md))
- [ ] Deployed (if beyond prototype)

## References

- [architecture.md](references/architecture.md) -- Multi-agent design, success criteria, edge cases, constraints
- [agent-example.md](references/agent-example.md) -- Example agent.py and retrievers.py
- [audio-integration.md](references/audio-integration.md) -- Voice interface with Gemini Live
- [evaluation-guide.md](references/evaluation-guide.md) -- Eval metrics and methodology
- [ingestion-scripts.md](references/ingestion-scripts.md) -- Detailed script docs
- [terraform-example.md](references/terraform-example.md) -- Example Terraform config
