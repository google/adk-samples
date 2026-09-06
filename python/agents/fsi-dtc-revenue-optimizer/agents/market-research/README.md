# Market Research Agent

A standalone ADK agent that performs deep web research using Gemini Deep Research to produce detailed, cited reports on market trends, competitive landscapes, and consumer sentiment.

## Overview

This agent wraps the [Gemini Deep Research](https://ai.google.dev/gemini-api/docs/deep_research) Interactions API. When invoked, it:

1. Accepts a research query (e.g., "current interior design trends 2026–2027")
2. Calls the Deep Research API, which autonomously plans, searches the web, reads sources, and synthesizes findings
3. Returns a comprehensive, cited research report

The agent exposes an A2A (Agent-to-Agent) protocol interface, allowing the orchestrator agent to call it over HTTP.

## Quick Start

```bash
# Install dependencies
make install

# Start local server (port 8002 recommended when running alongside orchestrator + product-strategy)
make local-backend PORT=8002

# Or use the ADK playground
make playground
```

## Authentication & Configuration

This agent uses **two separate authentication paths** because the Deep Research
Interactions API is only available through AI Studio, not Vertex AI:

| Component | Auth Method | Backend | Purpose |
|---|---|---|---|
| Root Agent (LLM) | Application Default Credentials (ADC) | **Vertex AI** | Normal conversation & orchestration |
| `deep_research` tool | `GEMINI_API_KEY` (API key) | **AI Studio** (non-Vertex) | Gemini Deep Research Interactions API |

### Vertex AI — Root Agent

The root agent's LLM (e.g. `gemini-3-flash-preview`) runs on **Vertex AI**.
Authentication is handled automatically via
[Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials)
(`google.auth.default()`). The environment variable `GOOGLE_GENAI_USE_VERTEXAI`
is set to `True` by default in `agent.py`.

No extra setup is needed beyond having valid ADC configured (e.g. via
`gcloud auth application-default login` locally, or a service account on
Cloud Run).

### AI Studio API Key — Deep Research Tool

The `deep_research` tool calls the
[Gemini Deep Research Interactions API](https://ai.google.dev/gemini-api/docs/deep_research),
which is currently only available through the **AI Studio** endpoint. It creates
a separate `genai.Client` with `vertexai=False` and authenticates using an
**API key** stored in the `GEMINI_API_KEY` environment variable.

#### How to obtain the API key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey).
2. Click **Create API key** and select your Google Cloud project.
3. Copy the generated key and set it as `GEMINI_API_KEY` in your `.env` file.

> **Note:** This is an *AI Studio* API key, not a GCP Console API key. Make sure
> the key is associated with a project that has the Gemini API enabled.

### Environment Variables

Copy `.env.sample` to `.env` and fill in the values:

| Variable | Description |
|----------|-------------|
| `ADK_MODEL` | Model ID for the LLM agent (default: `gemini-3-flash-preview`) |
| `GOOGLE_CLOUD_PROJECT` | Your Google Cloud project ID (auto-detected from ADC if unset) |
| `GOOGLE_CLOUD_LOCATION` | API location (default: `global`) |
| `GEMINI_API_KEY` | AI Studio API key for the Deep Research Interactions API ([get one here](https://aistudio.google.com/apikey)) |
| `LOGS_BUCKET_NAME` | GCS bucket for telemetry and GenAI logs. |

## Architecture

```
agents/market-research/
├── app/
│   ├── __init__.py
│   ├── agent.py            # Core agent with deep_research tool
│   ├── fast_api_app.py     # A2A FastAPI server
│   ├── prompts/
│   │   ├── system_prompt.md
│   │   └── system_instructions.md
│   └── app_utils/
│       ├── telemetry.py
│       └── typing.py
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── Makefile
└── .env.sample
```

## Deployment

```bash
# Deploy to Cloud Run
make deploy

# Deploy to Agent Engine
make deploy-agent-engine
```

## Development

```bash
# Run linting
make lint

# Run tests
make test

# Run evaluations
make eval
```
