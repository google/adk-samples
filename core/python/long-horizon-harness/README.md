<div align="center">
  <h1>Long Horizon</h1>
  <h3>A reference implementation of an agent harness on ADK and Google's Agent Platform.</h3>
</div>

Long Horizon is a **reference implementation**: a complete, running agent that shows how to build the harness around an ADK agent — cross-session memory, a per-user sandbox, tool guardrails, sub-agents, and a self-improvement loop. Read it, then lift the patterns into your own agent. It's a reference sample for demonstrative purposes only, _not an officially supported Google product_ (no SLA or support). Review the [security model](docs/security-model.md) before pointing it at anything real.

> **Study the custom interfaces** → [`AGENTS.md`](AGENTS.md) (each row links to the implementing function) · **Run it yourself** → [Quickstart](#quickstart) · **Architecture** → [`docs/architecture.md`](docs/architecture.md)

**Features:**

*Memory & self-improvement*

- **Cross-session memory** — Memory Bank is the only durable store, surfaced each turn via `PreloadMemoryTool`.
- **Background self-improvement** — after every turn a fire-and-forget judge writes facts to memory and techniques to the skill library, never blocking the response.
- **It dreams** — a nightly cross-session pass surfaces patterns one conversation can't (mentioned Monday, surfaced Friday) and consolidates your memories. See [`docs/memory.md`](docs/memory.md).

*Sandbox & secrets*

- **Per-user sandbox** — a JWT-routed BYOC container on Agent Runtime, warm between turns and reattached on the next message (snapshot survival opt-in). See [`docs/sandbox-lifecycle.md`](docs/sandbox-lifecycle.md).
- **Bring-your-own secrets** — API keys live per-user in Secret Manager and inject into sandbox commands; the model sees the name, never the value. Bulk-import via `.env` upload.
- **Connect Google** — one-click OAuth wires `gcloud`/`bq` and the `gws` Workspace CLI into your sandbox; tokens are stored as per-user secrets, never shown to the model. Bring your own OAuth client — [setup](docs/quickstart.md#6-connect-google-optional-oauth).

*Integration & extensibility*

- **A2A-native** — any agent or script drives Horizon over A2A (agent-to-agent), and it streams structured UI parts back over the same channel. No custom frontend per feature.
- **Sub-agents** — blocking `delegate` and fire-and-forget `agent`, each with its own context window and toolset.
- **Skills & custom scripts** — drop a `SKILL.md` or a `scripts/<name>.py` in the workspace and `/reload` picks it up mid-session. No fork, no redeploy.
- **Reminders & scheduled chats** — reminders fire as real, persisted chats in a "Scheduled" folder, viewable and replayable in the web UI.

*Reliability & safety*

- **Resumability + compaction** — sessions resume cleanly; `HorizonSummarizer` compresses old turns so context stays focused on what's current.
- **Guardrails** — iteration-budget, no-progress, and repeated-failure halts share one `halt_reason` and reset together at the turn boundary.
- **Self-reporting** — the agent files structured reports to maintainers (HITL-gated), and users can send feedback.

## The stack

| Layer | What | Why managed |
|---|---|---|
| **Runner** | ADK 2.5.x — plugins, callback graph, resumability, events compaction | The agent loop |
| **Model** | Gemini 3.6 Flash (root, default) · Flash Latest (web research) | First-party Vertex routing, no quota plumbing |
| **Memory** | Memory Bank (managed store) **+** a custom self-improvement loop (per-turn judge fork, pre-compaction flush, nightly dream-review consolidation) — see [`docs/memory.md`](docs/memory.md) | Cross-session semantic memory without owning a vector store |
| **Sessions** | Agent Runtime sessions | Persisted session/state without running your own store |
| **Sandbox** | Agent Runtime (BYOC container) | One per user, JWT-routed; reattached between turns (snapshot survival opt-in) |
| **Frontend** | Vite 8 + TanStack Router/Query + React 18 | UI surfaces stream as JSON over A2A |
| **Surface** | Cloud Run + Cloud SQL + Cloud Scheduler | Cloud Run scales to zero between turns (Cloud SQL bills continuously) |

Custom code earns its keep in a handful of **interfaces** on top of the managed primitives: the tool-guardrail chain, the `Environment` ContextVar sandbox layer, the per-user secret store, the dynamic delegate + HITL resurfacing, the three-tier system-prompt assembler, and the self-improvement loop layered on managed Memory Bank (per-turn judge fork, pre-compaction flush, nightly dream-review; see [`docs/memory.md`](docs/memory.md)). Compaction, resumability, and prefix caching are ADK/Vertex knobs Horizon only configures; routines and the scheduler are applications composed from the interfaces. (Those interfaces are the rows in [`AGENTS.md`](AGENTS.md).)

Full subsystem walkthrough: [`docs/architecture.md`](docs/architecture.md).

---

## Quickstart

**Run it locally.** For the step-by-step, see [`docs/quickstart.md`](docs/quickstart.md).

### Prerequisites

[uv](https://docs.astral.sh/uv/getting-started/installation/), [google-agents-cli](https://pypi.org/project/google-agents-cli/) (`uv tool install google-agents-cli` — provides the `agents-cli` command), [Google Cloud SDK](https://cloud.google.com/sdk/docs/install), `make`, and Node.js 20.19+ or 22.12+ (Vite 8 — web UI only). On Windows, use WSL.

### Setup and run

```bash
# 1. Clone and enter this sample
git clone https://github.com/google/adk-samples.git
cd adk-samples/core/python/long-horizon-harness

# 2. GCP access for Vertex inference (needs a project with BILLING enabled)
gcloud auth application-default login
gcloud config set project <your-project-id>          # or leave your gcloud default
gcloud services enable aiplatform.googleapis.com     # the Vertex AI API

# 3. Run — the first run installs deps and seeds .env from .env.example
make dev-local                                 # backend :8001 + web UI :3000
```

Open <http://localhost:3000>, or run one-shot from the terminal with `agents-cli run "your prompt"`. To install without starting servers: `make setup`.

> **Cost:** the test suites (`tests/unit` / `tests/integration`) run free with no GCP. Everything else calls **Vertex, which is billed per token**, so you need a project with billing enabled. `make deploy` additionally stands up always-on resources (Cloud SQL, Cloud Scheduler) — see [Deploy](#deploy) for teardown.

`make dev-local` is the simplest way to start: tools run on your host, sessions stay in-process, and no Agent Platform sandbox, Agent Runtime, or Cloud SQL is provisioned. The trade-off is **no cross-session memory** and **no isolated sandbox** (tools run directly on your machine); inference still calls Vertex (there is no local model — `local` only moves **tool execution** to your host). If `GOOGLE_CLOUD_PROJECT` is blank in `.env`, `make dev` falls back to your active `gcloud` project; switch to the Agent Platform sandbox with `LHA_ENVIRONMENT_BACKEND=sandbox` (or `make dev-sandbox`).

```bash
uv run pytest tests/unit tests/integration   # deterministic — no GCP needed
agents-cli eval run                          # LLM-behavior evalsets
```

Full config reference: [`docs/configuration.md`](docs/configuration.md).

---

## Learn & adapt

Horizon is a sample — **configure it with environment variables and adapt it by editing the code** (there's no wrapper API). Where to go next:

- **The custom interfaces** → [`AGENTS.md`](AGENTS.md) — each row links to the function that implements it.
- **Architecture** → [`docs/architecture.md`](docs/architecture.md) — the map + per-subsystem start-here files (and the runtime diagram).
- **Adapt / extend** → [`docs/extending.md`](docs/extending.md) — swap the model, add tools/routes, plug a custom sandbox, teach it skills.
- **Configuration** → [`docs/configuration.md`](docs/configuration.md) — every environment variable + dependency extra.
- **Security** → [`docs/security-model.md`](docs/security-model.md) — Horizon trusts the LLM within tool- and sandbox-level guardrails (identity, the guard chain, what each route exposes). Review it before pointing it at anything real.

## Deploy

`make deploy` is **one command** that provisions infrastructure *and* ships code, in three steps:

1. **Terraform** (`terraform/`) provisions the foundation (Cloud SQL, Secret Manager, IAM service accounts + roles, Cloud Scheduler jobs, IAP, GCS buckets, required-API enablement), plus the two Cloud Run **skeletons** (`lha` backend, `lha-web` IAP proxy) with placeholder images.
2. **`agents-cli deploy`** builds + rolls the real backend image onto `lha`.
3. **Cloud Build + `gcloud run deploy`** builds + rolls the web image onto `lha-web`.

Both services `ignore_changes` on their image, so steps 2–3 never fight Terraform. Prereqs: `terraform`, `agents-cli`, `gcloud`, and a project (`gcloud config set project <id>` or `PROJECT_ID=<id> make deploy`).

> **IAP access is empty by default.** Set `TF_VAR_iap_users='["user:you@example.com"]'` before `make deploy` (or re-run after) or you'll be locked out of the web UI.

**Billable, always-on resources** (Cloud SQL especially) — tear everything down when you're done:

```bash
make destroy   # flips the delete guards off, then terraform destroy of all the above
```

## Disclaimer

This repository is for demonstrative purposes only and is **not an officially supported Google product**. It is reference/sample code provided **without warranty or support of any kind**; review, test, and secure it before any real use. Licensed under Apache 2.0 (see [`LICENSE`](LICENSE)).
