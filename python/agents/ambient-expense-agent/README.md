# Ambient Expense Agent

A production-ready **ambient agent** that processes expense reports arriving via
Pub/Sub and routes them through an **ADK 2.0 graph-based workflow**. Low-value
expenses are auto-approved instantly; high-value ones go through LLM risk
analysis and **human-in-the-loop approval** before a decision is made.

<table>
  <thead>
    <tr>
      <th colspan="2">Key Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🔄</td>
      <td><strong>ADK 2.0 Graph Workflow:</strong> Conditional routing with function nodes and LLM agents in the same graph — business rules stay in code, LLM handles judgment calls.</td>
    </tr>
    <tr>
      <td>📡</td>
      <td><strong>Ambient & Event-Driven:</strong> Listens for expense events via <a href="https://cloud.google.com/pubsub">Pub/Sub</a> triggers and processes them automatically in the background.</td>
    </tr>
    <tr>
      <td>✋</td>
      <td><strong>Human-in-the-Loop:</strong> High-value expenses pause the workflow with <code>RequestInput</code> until a manager approves or rejects via a dedicated approval UI.</td>
    </tr>
    <tr>
      <td>☁️</td>
      <td><strong>Agent Runtime Deployment:</strong> The agent backend runs on <a href="https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview">Agent Runtime</a> (Vertex AI). A Cloud Run approval UI and Pub/Sub trigger are provisioned via Terraform.</td>
    </tr>
  </tbody>
</table>

| Attribute | Description |
| :--- | :--- |
| **Interaction Type** | Ambient (event-driven) with HITL approval |
| **Complexity** | Intermediate |
| **Agent Type** | ADK 2.0 Graph-based Workflow |
| **Trigger Sources** | Pub/Sub push → Agent Runtime API passthrough |
| **Deployment** | Agent Runtime (backend) + Cloud Run (approval UI) |

## How It Works

The agent is built as an ADK 2.0 [`Workflow`](https://adk.dev/workflows/) with
conditional routing. The $100 threshold lives in code, not in a prompt — only
high-value expenses hit the LLM.

```
  Expense arrives (Cloud Scheduler or direct publish)
            │
       Pub/Sub topic
            │
   Push subscription (OIDC auth)
            │
  Agent Runtime API passthrough
    /api/apps/expense_agent/trigger/pubsub
            │
     parse & extract data
            │
      route by amount
       │          │
   < $100       >= $100
       │          │
  auto-approve   LLM reviews risk
   (done)        & emits alert log
                  │
            manager receives
             email alert
                  │
            manager approves
             or rejects
             (approval UI)
                  │
            agent logs decision
             & resumes workflow
```

### Deployment Architecture

```
Cloud Scheduler (optional cron)
        │
        ▼
  Pub/Sub topic: expense-reports
        │
        ▼ push (OIDC, roles/aiplatform.user)
  Agent Runtime API passthrough
  → /api/apps/expense_agent/trigger/pubsub
        │
        ▼
  ADK trigger route decodes payload,
  creates session, runs workflow
        │
   (if >= $100)
        │
        ▼
  Cloud Monitoring alert → manager email
        │
        ▼
  Approval UI (Cloud Run, IAP-protected)
  → manager approves/rejects via POST /run
```

## Local Development

### 1. Prerequisites

- Python 3.11–3.12
- [uv](https://docs.astral.sh/uv/)
- A Google AI Studio key or Vertex AI project with ADC

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your project or API key
```

### 3. Install and run

Start the backend:

```bash
make install && make dev
```

In a separate terminal, start the approval UI:

```bash
make install-frontend && make dev-frontend
```

### 4. Try it out

Open the ADK playground to interact with the agent directly:

```bash
make playground
```

This starts the ADK web UI at `http://localhost:8501`.

To test the full Pub/Sub trigger flow, send an expense in another terminal:

```bash
curl -s http://localhost:8080/apps/expense_agent/trigger/pubsub \
  -H "Content-Type: application/json" \
  -d "{\"message\":{\"data\":\"$(echo '{"amount":250,"submitter":"alice@company.com","category":"travel","description":"Flight to NYC","date":"2026-04-10"}' | base64)\",\"attributes\":{\"source\":\"test\"}},\"subscription\":\"test-sub\"}"
```

This $250 expense triggers review + HITL approval. Open the approval UI
at `http://localhost:8081/approval` to approve or reject it.

> **Tip:** Expenses under $100 are auto-approved — change `amount` to
> `45` to test that path.

## Cloud Deployment

Deploying the agent uses a two-step process:

1. **`agents-cli deploy`** — packages and deploys the agent to Agent Runtime.
2. **`terraform apply`** — creates the Pub/Sub subscription, Cloud Monitoring alerts, IAM, and the frontend Cloud Run service.

The `Makefile` handles both steps in sequence.

**Prerequisites:**
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- [agents-cli](https://goo.gle/agents-cli) — `pip install google-agents-cli`
- [Terraform](https://www.terraform.io/)
- [uv](https://docs.astral.sh/uv/)

```bash
gcloud config set project YOUR_PROJECT_ID
make deploy NOTIFICATION_EMAIL=finance@example.com
```

This will:
1. Deploy the agent to Agent Runtime via `agents-cli deploy`
2. Build and push the frontend container image to Artifact Registry
3. Apply Terraform: Pub/Sub topic + authenticated push subscription,
   Cloud Monitoring alert, IAM bindings, and the Cloud Run approval UI

> **Note:** IAP on the approval UI can take **5–10 minutes** to propagate after
> initial deployment. If you see a `403 Forbidden`, wait a few minutes and refresh.

### Region

The Makefile defaults to `us-east1`. To use a different region:

```bash
make deploy REGION=us-central1 NOTIFICATION_EMAIL=finance@example.com
```

### Test the deployed agent

```bash
make remote-test
```

This publishes a $250 travel expense to the `expense-reports` topic. The agent
will route it to the review path, analyze risk, email an alert to
`NOTIFICATION_EMAIL`, and pause for human approval. Open the approval UI
(URL printed by `make deploy`) to approve or reject.

### Cleanup

```bash
make clean NOTIFICATION_EMAIL=finance@example.com
```

This tears down the Pub/Sub, monitoring, IAM, and frontend Cloud Run resources
via Terraform, then deletes the Agent Runtime engine.

## How the Auth Works (Pub/Sub → Agent Runtime)

The Pub/Sub push subscription uses OIDC authentication to call the
Agent Runtime API passthrough:

```
Pub/Sub push → OIDC token (audience: https://{REGION}-aiplatform.googleapis.com/)
             → invoker SA (roles/aiplatform.user)
             → Agent Runtime API: /api/apps/expense_agent/trigger/pubsub
```

This is different from Cloud Run push subscriptions (which use `roles/run.invoker`
and an audience matching the service URL). The `iam.tf` sets this up automatically.

## Customization

| What to change | How |
| --- | --- |
| **Approval threshold** | Change `review_threshold` in `expense_agent/config.py` |
| **LLM model** | Change `model` in `expense_agent/config.py` |
| **Expense schema** | Edit the `ExpenseData` Pydantic model in `expense_agent/agent.py` |
| **Review logic** | Edit the `review_agent` instruction in `expense_agent/agent.py` |
| **Approval UI** | Edit `frontend/static/approval.html` |
| **Downstream actions** | Add workflow nodes for Slack, databases, or notifications |
| **Add Cloud Scheduler** | Create a Cloud Scheduler job that publishes to `expense-reports` topic on a cron schedule |

## Troubleshooting

- For general ADK issues, see the [ADK documentation](https://adk.dev).
- For Agent Runtime logs, check Cloud Logging with resource type `aiplatform.googleapis.com/ReasoningEngine`.
- For trigger endpoint details, see [Ambient Agents](https://adk.dev/runtime/ambient-agents/).
- For Agent Runtime deployment, see [Deploy to Agent Runtime](https://adk.dev/deploy/agent-engine/).

## Disclaimer

This agent sample is provided for illustrative purposes only. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents.

Users are solely responsible for any further development, testing, security hardening, and deployment of agents based on this sample. We recommend thorough review, testing, and the implementation of appropriate safeguards before using any derived agent in a live or critical system.
