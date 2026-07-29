# Retail Product Search

Semantic product search agent on Google Cloud (Vertex AI Vector Search,
BigQuery, embeddings). Use to build e-commerce search, catalog discovery, or
shopping assistant agents.

## Install

```bash
npx skills add tanvisinghal-0105/solution_skills --skill retail-product-search
```

Hosts that support the skills spec (Claude Code, Gemini CLI, Codex, ...)
auto-detect and drop `SKILL.md` into their skills directory
(e.g. `~/.claude/skills/`, `~/.agents/skills/`).

## Prerequisites

- Python 3.10+
- [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) with ADC
  configured (`gcloud auth application-default login`)
- A GCP project with billing enabled and BigQuery + Vertex AI APIs on:
  ```bash
  gcloud services enable bigquery.googleapis.com aiplatform.googleapis.com
  ```

## Run

In a fresh workspace, launch your AI coding agent and trigger the skill:

```
Use the retail-product-search skill to set up a product search agent on Google Cloud.
```

The agent walks Q-MODE (2 questions Quick / 4 questions Full), runs
`scripts/bootstrap.sh` to create the venv, then `scripts/setup.py` to validate
the catalog, ingest to BigQuery, create the Vector Search collection, and
launch the ADK web UI.

## Use your own catalog

Choose Full setup and point Q-B at a local CSV or `gs://` URI. CSV needs
`product_id`, `name`, `price`, `description`. Optional: `category`, `brand`,
`image_url`, `rating`, `stock`.

## Cleanup

In the agent chat:

```
clean up the GCP resources
```

Runs `cleanup.py --confirm` to delete the BigQuery dataset and Vector Search
collection.

## Troubleshooting

| Error | Fix |
|---|---|
| `MethodNotImplemented: 501` from Vector Search | `VECTOR_SEARCH_COLLECTION` has a newline. Re-export on one line |
| `ModuleNotFoundError: google.adk` | `bash -c "pip install -e '${SKILL_DIR}[adk]'"` (the `bash -c` matters in zsh) |
| `Package requires Python: 3.9.X` | Recreate venv with `python3.12 -m venv .venv` |
| `BILLING_DISABLED` / `PERMISSION_DENIED` | GCP project setup — see [references/troubleshooting.md](references/troubleshooting.md) |

Full table: [references/troubleshooting.md](references/troubleshooting.md).

## What gets built

- BigQuery dataset `retail_skill_products.products`
- Vertex AI Vector Search collection `retail-skill-products-collection` in
  `us-central1`, with auto-embeddings via `gemini-embedding-001`
- Workspace venv with the skill installed editable + a `design-spec.md`

The skill's source code stays in the install directory — nothing is copied
to your workspace.

## License

Apache 2.0
