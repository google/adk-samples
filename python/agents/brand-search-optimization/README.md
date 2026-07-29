# Brand Search Optimization

This sample is an Agent Development Kit (ADK) multi-agent workflow that helps optimize retail product titles for brand search performance.

## What This Agent Does

- Finds high-value keywords for a brand from product catalog data in BigQuery.
- Searches a target website using browser tooling.
- Compares top search results with your product data.
- Recommends title improvements to reduce zero/low-result search outcomes.

## Architecture

The root agent routes work to specialized sub-agents:

- `keyword_finding`: extracts relevant brand keywords.
- `search_results`: navigates and analyzes search result pages.
- `comparison`: compares candidate titles and proposes improvements.

## Prerequisites

- Python 3.12+
- `uv` installed: https://docs.astral.sh/uv/
- Google Cloud project access
- Application Default Credentials:

```bash
gcloud auth application-default login
```

## Setup

1. Clone the repository and open this agent directory:

```bash
git clone https://github.com/google/adk-samples.git
cd adk-samples/python/agents/brand-search-optimization
```

2. Create your environment file:

```bash
cp .env.example .env
```

3. Sync dependencies:

```bash
uv sync --dev
```

4. (Optional) Populate sample BigQuery data:

```bash
uv run python -m deployment.bq_populate_data
```

## Run The Agent

CLI mode:

```bash
uv run adk run brand_search_optimization
```

Web UI mode:

```bash
uv run adk web
```

Then select `brand-search-optimization` from the app dropdown.

## Evaluation

```bash
uv run adk eval brand_search_optimization eval/data/eval_data1.evalset.json --config_file_path eval/data/test_config.json
```

## Tests, Lint, and Type Checking

Run tests with warnings enabled:

```bash
uv run pytest -s -W default
```

Run Ruff checks and formatting:

```bash
uv run ruff check . --fix
uv run ruff format .
```

Run mypy:

```bash
uv run mypy .
```

## Deployment

```bash
uv sync --group deployment
uv run python deployment/deploy.py --create
```

For a post-deploy validation flow, see `deployment/test_deployment.py`.

## Configuration

Environment variables are documented in `.env.example`.

Important variables:

- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `GOOGLE_GENAI_USE_VERTEXAI`
- `MODEL`
- `DATASET_ID`
- `TABLE_ID`
- `DISABLE_WEB_DRIVER`
- `STAGING_BUCKET`

## Example Interaction

See `tests/example_interaction.md` for a full sample session.

## Disclaimer

This sample is for educational and prototyping use. It is not production hardened and should be reviewed, tested, and secured before production deployment.