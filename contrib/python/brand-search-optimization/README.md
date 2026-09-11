# Brand Search Optimization

This recipe is an Agent Development Kit (ADK) multi-agent workflow that helps e-commerce brands optimize product titles for search visibility and recover zero/low-result retail queries using BigQuery and Gemini Computer Use.

## What This Recipe Does

- **Extracts Brand Catalog Data**: Queries product titles, descriptions, and attributes from BigQuery with typed tool parameters.
- **Identifies High-Intent Search Terms**: Mines high-value shopper keywords (category, style, attribute, use-case) to find what shoppers actually search for.
- **Visual Retail Search via Computer Use**: Navigates retail search engines using **Gemini Computer Use** (powered by Playwright and ADK `ComputerUseToolset`), capturing screenshots and observing how top organic competitor products structure their titles.
- **Structured Title Evaluation & Scoring**: Evaluates keyword gaps, calculates searchability scores (0-100), and generates structured, high-converting product title recommendations.

## Architecture

The workflow coordinates specialized agents:

1. `keyword_finding_agent`: queries BigQuery product catalog data and ranks high-intent search keywords.
2. `search_results_agent`: visual browser agent using **Gemini Computer Use** to navigate retail search engines and observe competitor title structures.
3. `comparison_root_agent`: coordinates generation and critique of title optimizations to deliver a structured `TitleOptimizationReport`.

## Prerequisites

- Python 3.11 - 3.12
- `uv` installed: https://docs.astral.sh/uv/
- Google Cloud project with Vertex AI and BigQuery access
- Application Default Credentials:

```bash
gcloud auth application-default login
```

## Setup

1. Clone the repository and navigate to this recipe directory:

```bash
git clone https://github.com/google/adk-samples.git
cd adk-samples/contrib/python/brand-search-optimization
```

2. Create your environment file:

```bash
cp .env.example .env
```

3. Sync dependencies and install Playwright browser binaries:

```bash
uv sync --dev
uv run playwright install chromium
```

4. (Optional) Populate sample BigQuery catalog data:

```bash
uv run python -m deployment.bq_populate_data
```

## Run The Agent

### CLI Mode

```bash
uv run adk run brand_search_optimization
```

### Web UI Mode

```bash
uv run adk web
```

Then select `brand_search_optimization` from the application dropdown.

## Evaluation

Run the evaluation suite:

```bash
uv run adk eval brand_search_optimization eval/data/eval_data1.evalset.json --config_file_path eval/data/test_config.json
```

## Tests, Lint, and Type Checking

Run unit and runnability tests:

```bash
uv run pytest -v
```

Run Ruff linting and formatting:

```bash
uv run ruff check . --fix
uv run ruff format .
```

Run type checking:

```bash
uv run mypy .
```

## Deployment

Deploy the agent to Vertex AI Agent Engine:

```bash
uv sync --group deployment
uv run python deployment/deploy.py --create
```

For post-deployment session testing, see `deployment/test_deployment.py`.

## Configuration

Environment variables are declared in `.env.example`:

- `GOOGLE_GENAI_USE_VERTEXAI`: Set to `1` for Vertex AI backend, `0` for Google AI Studio
- `GOOGLE_API_KEY`: Google AI Studio API key (when using AI Studio backend)
- `GOOGLE_CLOUD_PROJECT`: Google Cloud project ID
- `GOOGLE_CLOUD_LOCATION`: Vertex AI location (e.g., `us-central1` or `global`)
- `MODEL`: Model name (e.g., `gemini-3.5-flash`)
- `DATASET_ID`: BigQuery dataset ID (default: `products_data_agent`)
- `TABLE_ID`: BigQuery table ID (default: `shoe_items`)
- `DISABLE_WEB_DRIVER`: Set to `1` to run in offline/mock browser mode for headless testing (default: `0`)
- `STAGING_BUCKET`: GCS staging bucket for Cloud deployment
- `AGENT_VERSION`: Version string advertised in A2A agent card (default: `0.1.0`)
- `ALLOW_ORIGINS`: Allowed CORS origins for FastAPI server (comma-separated)
- `APP_URL`: Base URL advertised in A2A agent card (default: `http://0.0.0.0:8080`)
- `GOOGLE_CLOUD_AGENT_ENGINE_ID`: Agent Engine resource ID for remote session service
- `GOOGLE_CLOUD_AGENT_ENGINE_LOCATION`: Agent Engine location/region (e.g., `us-central1`)
- `LOGS_BUCKET_NAME`: GCS bucket for remote artifact storage
- `SESSION_SERVICE_URI`: URI for ADK session service (e.g., `shared://session`)
- `ARTIFACT_SERVICE_URI`: URI for ADK artifact service (e.g., `shared://artifact`)
- `HOST`: Host interface binding for FastAPI server (default: `0.0.0.0`)
- `PORT`: HTTP port for FastAPI server (default: `8080`)

## Example Interaction

See `tests/example_interaction.md` for a complete example interaction trace.

## Disclaimer

This recipe is for educational and prototyping use. It is not production hardened and should be reviewed, tested, and secured before production deployment.
