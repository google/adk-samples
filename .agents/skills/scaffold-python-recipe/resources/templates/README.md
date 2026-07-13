# <RECIPE_NAME>

This is a simple agent using the ADK Python SDK to demonstrate its capabilities.

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager - [Install](https://docs.astral.sh/uv/getting-started/installation/)

## Quick Start

> **Note**: All commands below must be run from the recipe root directory (`<OUTPUT_DIRECTORY>/<RECIPE_NAME>/`).

1. Install required packages:
   ```bash
   uv sync
   ```

2. Set up environment variables:
   Copy `.env.example` to `.env` and uncomment/configure the variables you need (like `GEMINI_API_KEY`, `GOOGLE_CLOUD_PROJECT`, etc.):
   ```bash
   cp .env.example .env
   ```

3. Test the agent in the command line (interactive mode):
   ```bash
   uv run adk run app
   ```

4. Or start the local FastAPI web server:
   ```bash
   uv run uvicorn app.fast_api_app:app --reload
   ```

## Running Tests

To run the unit, integration, and runnability tests:

```bash
uv run pytest
```

Or to run specific test suites:

```bash
# Run unit and runnability tests only
uv run pytest tests/unit

# Run integration tests only
uv run pytest tests/integration
```

## Commands

| Command | Description |
| ------- | ----------- |
| `uv run adk run app` | Run the agent in interactive CLI mode |
| `uv run uvicorn app.fast_api_app:app --reload` | Start the local FastAPI development server |
| `uv run pytest` | Run all test suites |
