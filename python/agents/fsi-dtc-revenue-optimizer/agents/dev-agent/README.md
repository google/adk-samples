# dev-agent

ReAct agent with A2A protocol [experimental]
Agent generated with [`googleCloudPlatform/agent-starter-pack`](https://github.com/GoogleCloudPlatform/agent-starter-pack) version `0.38.0`

## Project Structure

```
dev-agent/
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

| Command                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| `make install`             | Install dependencies using uv                                          |
| `make playground`          | Launch local development environment                                   |
| `make lint`                | Run code quality checks                                                |
| `make test`                | Run unit and integration tests                                         |
| `make deploy`              | Deploy agent to Cloud Run                                              |
| `make create-secrets`      | Create/update secrets in Secret Manager from `.env`                    |
| `make grant-secret-access` | Grant Cloud Run service account access to secrets                      |
| `make local-backend`       | Launch local development server with hot-reload                        |
| `make inspector`           | Launch A2A Protocol Inspector                                          |

For full command options and usage, refer to the [Makefile](Makefile).

## 🛠️ Project Management

| Command | What It Does |
|---------|--------------|
| `uvx agent-starter-pack enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `uvx agent-starter-pack setup-cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `uvx agent-starter-pack upgrade` | Auto-upgrade to latest version while preserving customizations |
| `uvx agent-starter-pack extract` | Extract minimal, shareable version of your agent |

---

## Configuration

### Environment Variables

Copy `.env.sample` to `.env` and fill in the values:

| Variable | Description |
|----------|-------------|
| `SKIP_JIRA` | Set to `"true"` to skip Jira and use a lightweight GCS-based task system instead. Defaults to `"false"`. |
| `SKIP_CHAT` | Set to `"true"` to skip sending Google Chat notifications. Defaults to `"false"`. |
| `ADK_MODEL` | The Gemini model to use (e.g. `gemini-3-flash-preview`). |
| `GOOGLE_CLOUD_PROJECT` | Your GCP project ID. |
| `GOOGLE_CLOUD_LOCATION` | GCP region (e.g. `global`). |
| `ASSET_BUCKET_NAME` | GCS bucket for task JSON files. |
| `VEO_GCS_BUCKET` | GCS bucket for campaign videos. |
| `LOGS_BUCKET_NAME` | GCS bucket for telemetry and GenAI logs. |
| `JIRA_BASE_URL` | Jira Cloud base URL (e.g. `https://your-org.atlassian.net`). Not needed when `SKIP_JIRA=true`. |
| `JIRA_USER_EMAIL` | Jira user email for API auth and ticket assignment. |
| `JIRA_API_TOKEN` | Jira API token (secret). |
| `GOOGLE_CHAT_WEBHOOK_URL` | Google Chat webhook URL for notifications (secret). Not needed when `SKIP_CHAT=true`. |

### SKIP_JIRA Mode

When `SKIP_JIRA="true"`, the agent replaces all Jira operations with a GCS-based task system:

- **Create task**: Generates a unique task ID (e.g. `TASK-a3f7b2c1`), extracts video URLs from the description, and uploads a JSON file to `gs://{ASSET_BUCKET_NAME}/tasks/{task_id}.json`.
- **Get task**: Downloads the task JSON from GCS and displays all details including video URLs.
- **Start task**: Updates the task JSON with status "In Progress", assignee, and start date.
- **Get videos**: Retrieves video URLs from the task JSON. Falls back to listing videos from the bucket specified by `VEO_GCS_BUCKET`.

Chat notifications use the task ID instead of Jira ticket keys, and no Jira info is shown.

### Google Chat Webhook

To send notifications to a Google Chat space, create an incoming webhook:

1. Follow the [Google Chat webhook guide](https://developers.google.com/workspace/chat/quickstart/webhooks) to create a webhook in your Chat space.
2. Copy the webhook URL and set it as `GOOGLE_CHAT_WEBHOOK_URL` in your `.env`.

## Development

Edit your agent logic in `app/agent.py` and test with `make playground` - it auto-reloads on save.
See the [development guide](https://googlecloudplatform.github.io/agent-starter-pack/guide/development-guide) for the full workflow.

## Deployment

```bash
gcloud config set project <your-project-id>
make deploy
```

### Secrets

Sensitive environment variables (`GOOGLE_CHAT_WEBHOOK_URL`, `JIRA_API_TOKEN`) are stored in [Secret Manager](https://cloud.google.com/secret-manager) and mounted into the Cloud Run revision at deploy time. They are **not** passed as plain-text environment variables.

**First-time setup** (before the first `make deploy`):

```bash
# Push secret values from your local .env to Secret Manager
make create-secrets
```

`make deploy` automatically runs `make grant-secret-access` to ensure the Cloud Run service account can read the secrets.

To update a secret value later, edit `.env` and re-run `make create-secrets`. The next `make deploy` will pick up the latest version automatically.

> 💡 For local development, the agent reads these values directly from `.env` — no Secret Manager access is needed.

To add CI/CD and Terraform, run `uvx agent-starter-pack enhance`.
To set up your production infrastructure, run `uvx agent-starter-pack setup-cicd`.
See the [deployment guide](https://googlecloudplatform.github.io/agent-starter-pack/guide/deployment) for details.

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging.
See the [observability guide](https://googlecloudplatform.github.io/agent-starter-pack/guide/observability) for queries and dashboards.

## A2A Inspector

This agent supports the [A2A Protocol](https://a2a-protocol.org/). Use `make inspector` to test interoperability.
See the [A2A Inspector docs](https://github.com/a2aproject/a2a-inspector) for details.
