# orchestrator

ReAct agent with A2A protocol [experimental]
Agent generated with [`googleCloudPlatform/agent-starter-pack`](https://github.com/GoogleCloudPlatform/agent-starter-pack) version `0.38.0`

## Project Structure

```
orchestrator/
├── app/         # Core agent code
│   ├── agent.py               # Main agent logic
│   ├── fast_api_app.py        # FastAPI Backend server
│   └── app_utils/             # App utilities and helpers
├── tests/                     # Unit, integration, and load tests
├── GEMINI.md                  # AI-assisted development guide
├── Makefile                   # Development commands
└── pyproject.toml             # Project dependencies
```

> 💡 **Tip:** Use [Gemini CLI](https://github.com/google-gemini/gemini-cli) for AI-assisted development - project context is pre-configured in `GEMINI.md`.

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager (used for all dependency management in this project) - [Install](https://docs.astral.sh/uv/getting-started/installation/) ([add packages](https://docs.astral.sh/uv/concepts/dependencies/) with `uv add <package>`)
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)
- **make**: Build automation tool - [Install](https://www.gnu.org/software/make/) (pre-installed on most Unix-based systems)


## Quick Start

Install required packages and launch the local development environment:

```bash
make install && make playground
```

## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `make install`       | Install dependencies using uv                                                               |
| `make playground`    | Launch local development environment                                                        |
| `make lint`          | Run code quality checks                                                                     |
| `make test`          | Run unit and integration tests                                                              |
| `make deploy`        | Deploy agent to Cloud Run                                                                   |
| `make local-backend` | Launch local development server with hot-reload                                             |
| `make inspector`     | Launch A2A Protocol Inspector                                                               |

For full command options and usage, refer to the [Makefile](Makefile).

## 🛠️ Project Management

| Command | What It Does |
|---------|--------------|
| `uvx agent-starter-pack enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `uvx agent-starter-pack setup-cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `uvx agent-starter-pack upgrade` | Auto-upgrade to latest version while preserving customizations |
| `uvx agent-starter-pack extract` | Extract minimal, shareable version of your agent |

---

## Development

Edit your agent logic in `app/agent.py` and test with `make playground` - it auto-reloads on save.
See the [development guide](https://googlecloudplatform.github.io/agent-starter-pack/guide/development-guide) for the full workflow.

## Deployment

```bash
gcloud config set project <your-project-id>
make deploy
```

To add CI/CD and Terraform, run `uvx agent-starter-pack enhance`.
To set up your production infrastructure, run `uvx agent-starter-pack setup-cicd`.
See the [deployment guide](https://googlecloudplatform.github.io/agent-starter-pack/guide/deployment) for details.

## Configuration

### Environment Variables

Copy `.env.sample` to `.env` and fill in the values:

| Variable | Description |
|----------|-------------|
| `ADK_MODEL` | The Gemini model to use (e.g. `gemini-3-flash-preview`). |
| `MARKET_RESEARCH_AGENT_URL` | URL of the Market Research Agent. |
| `PRODUCT_STRATEGY_AGENT_URL` | URL of the Product Strategy Agent. |
| `LOGS_BUCKET_NAME` | GCS bucket for telemetry and GenAI logs. |
| `BQ_DATA_AGENT_PROJECT` | Google Cloud project ID that hosts the BigQuery data agent. |
| `BQ_DATA_AGENT_ID` | BigQuery Data agent ID. |
| `BQ_DATA_AGENT_LOCATION` | BigQuery Data agent location (defaults to `global`). |

### BQ Data Agent Configuration

The orchestrator queries a **BigQuery Conversational Analytics Data Agent** for internal data analysis (product catalog, inventory, dead stock, etc.).

#### Prerequisites

1. Enable the required APIs in your project:
   ```bash
   gcloud services enable geminidataanalytics.googleapis.com \
                          cloudaicompanion.googleapis.com
   ```
2. Create a data agent in the [BigQuery Agents Hub](https://console.cloud.google.com/bigquery/agents_hub) with your product catalog tables as knowledge sources, then **Publish** it.
3. Grant the orchestrator's service account the following roles:
   - `roles/geminidataanalytics.dataAgentUser`
   - `roles/geminidataanalytics.dataAgentStatelessUser`
   - `roles/bigquery.dataViewer` on the source tables

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging.
See the [observability guide](https://googlecloudplatform.github.io/agent-starter-pack/guide/observability) for queries and dashboards.

## A2A Inspector

This agent supports the [A2A Protocol](https://a2a-protocol.org/). Use `make inspector` to test interoperability.
See the [A2A Inspector docs](https://github.com/a2aproject/a2a-inspector) for details.
