# Cross-Session Memory — Vertex AI Memory Bank

## Intent

A clone-and-study ADK agent that **remembers user preferences and facts across
sessions** using **Vertex AI Memory Bank**. The agent itself is a deliberately
mundane demo (weather/time tools); the interesting part is the **Memory Bank
wiring** — how memories are **written** after each turn
(`generate_memories_callback` → `add_session_to_memory()`) and **recalled** at
the start of a later session (`PreloadMemoryTool`), a single shared
`memory_bank_config` that declares which topics to extract, and **two deploy
paths** (Agent Engine and Cloud Run) that both enable Memory Bank via a
`ReasoningEngineContextSpec`.

## When To Use

- The user wants an agent that persists **user preferences/facts across
  sessions** without building or maintaining a memory store — Memory Bank
  extracts and consolidates them for you.
- The user wants a worked example of ADK's memory hooks: `PreloadMemoryTool`
  (automatic recall) plus `after_agent_callback` / `add_session_to_memory`
  (automatic write).
- The user needs the **same agent code deployable to both Agent Engine (Agent
  Runtime) and Cloud Run**, with Memory Bank enabled either way.
- The user wants to see how **managed memory topics**
  (`USER_PERSONAL_INFO` / `USER_PREFERENCES` / `EXPLICIT_INSTRUCTIONS`) are
  configured and passed to the platform.

## Eval

- **Scenarios Path**: none — this recipe ships **no eval datasets** and **no
  `agents-cli` eval config**. Do not look for `tests/eval/`.
- **Tests** live in `tests/`:
  - `tests/test_runnability.py` — import-time check that `root_agent` and `app`
    are defined (patches `google.auth.default` so it runs credential-less).
  - `tests/unit/test_dummy.py` — pure logic tests for `get_weather` /
    `get_current_time` (no GCP, no LLM).
  - `tests/integration/test_agent.py` — `test_agent_has_memory_wired` is the
    **only test that asserts the memory wiring** (`after_agent_callback ==
    generate_memories_callback` and a `PreloadMemoryTool` in `tools`);
    `test_agent_stream` runs the agent through a `Runner` with a **mock LLM** and
    an `InMemoryMemoryService`.
  - `tests/integration/test_agent_engine_app.py` — `register_feedback` on
    `AgentEngineApp` (external services mocked via `INTEGRATION_TEST`).
  - `tests/integration/test_server_e2e.py` — spins up the FastAPI server
    (`USE_IN_MEMORY_SESSION=true`, Gemini patched by `_e2e_bootstrap`) and hits
    `/run_sse` + `/feedback`.
- **None of these prove cross-session recall end-to-end** (they all mock the LLM
  and use in-memory services). To add real eval, drop an `agents-cli` config +
  datasets under `tests/eval/` (as the RAG recipes do) and wire it into the
  `Makefile`; a meaningful memory eval needs a live Memory Bank instance
  (`--memory_service_uri=agentengine://…`) and multi-session scenarios.

## End-to-end flow

```
Session 1  (user tells the agent something)
  user: "I live in Austin."
   -> agent replies; after_agent_callback = generate_memories_callback
   -> callback_context.add_session_to_memory() ships the turn's events to
      VertexAiMemoryBankService
   -> Memory Bank extracts + consolidates facts under the configured topics
      (USER_PERSONAL_INFO / USER_PREFERENCES / EXPLICIT_INSTRUCTIONS)

Session 2  (a new session, later)
  user: "What's the weather?"
   -> PreloadMemoryTool queries Memory Bank at turn start and injects
      "lives in Austin" into the system instruction
   -> model answers for Austin without being reminded
```

Locally the memory service is `InMemoryMemoryService` (nothing persists); the
real `VertexAiMemoryBankService` is used only when `memory_service_uri` points at
an `agentengine://` instance (Cloud Run wires this automatically; `adk web`
needs the flag).

## Most interesting files to study (in order)

### Memory Bank config & deploy paths (the bulk of the logic)
1. **`app/app_utils/memory_config.py`** — the shared `memory_bank_config`, a
   `ReasoningEngineContextSpecMemoryBankConfig` whose `customization_configs`
   enable three **managed topics**: `USER_PERSONAL_INFO`, `USER_PREFERENCES`,
   `EXPLICIT_INSTRUCTIONS` (`KEY_CONVERSATION_DETAILS` is available but left
   off). This single object is the **one place you declare what Memory Bank
   extracts**, and it is consumed by **both** deploy paths. Custom topics are
   supported via `CustomMemoryTopic(label=..., description=...)`.
2. **`app/app_utils/deploy.py`** — the **Agent Engine** deploy CLI (`make
   deploy-agent-engine` runs it). The memory-relevant part: it wraps
   `memory_bank_config` in a `ReasoningEngineContextSpec` and passes it via
   `context_spec` in `AgentEngineConfig`, so the created/updated Agent Engine has
   Memory Bank enabled. A `click` command that also handles create-or-update by
   `display_name`, env vars/secrets, resource limits, and optional
   `--agent-identity` IAM setup.
3. **`app/fast_api_app.py`** — the **Cloud Run** entry point **and** its
   memory-enabling path. On startup it finds an existing Agent Engine by
   `display_name` or **creates one**, passing `memory_bank_config` via the same
   `context_spec`; then sets **both** `session_service_uri` and
   `memory_service_uri` to the same `agentengine://<resource>` URI and hands them
   to `get_fast_api_app()`. Setting `memory_service_uri` is what makes ADK use
   `VertexAiMemoryBankService` (so `PreloadMemoryTool` + the callback hit real
   Memory Bank). `USE_IN_MEMORY_SESSION=true` bypasses all of this for local dev.
4. **`app/agent_engine_app.py`** — the Agent Engine **runtime object**
   (`agent_engine`), an `AgentEngineApp(AdkApp)` wrapping `app.agent.app`. Adds
   telemetry/logging setup and a `register_feedback` operation. This is the
   `--entrypoint-object` that `deploy.py` deploys; note it does **not** re-declare
   Memory Bank — the config is applied at instance-creation time by `deploy.py`.

### Support
5. **`app/app_utils/telemetry.py`** — `setup_telemetry()`: OpenTelemetry / GenAI
   telemetry, optional prompt/response upload to GCS (`LOGS_BUCKET_NAME`, forced
   to `NO_CONTENT` metadata-only mode). No memory logic.
6. **`app/app_utils/typing.py`** — Pydantic `Request` / `Feedback` models used by
   the FastAPI `/feedback` endpoint and `register_feedback`.

### Agent (thin layer — read last)
7. **`app/agent.py`** — the actual agent, deliberately generic (`get_weather` /
   `get_current_time` demo tools). The two lines that matter: **`PreloadMemoryTool()`
   in `tools`** (recall — injects memories into the system instruction at turn
   start) and **`after_agent_callback=generate_memories_callback`** (write —
   `callback_context.add_session_to_memory()` ships the turn's events to Memory
   Bank). Model comes from the `MODEL_NAME` env var; exports
   `app = App(root_agent=root_agent, name="app")`. `app/__init__.py` bootstraps
   env (`load_dotenv`, ADC project discovery, default
   `GOOGLE_CLOUD_LOCATION=global`, `GOOGLE_GENAI_USE_VERTEXAI=True`).

## Data handling

- **No datastore of your own.** Memory Bank (a managed Vertex AI service living
  on an Agent Engine instance) stores and consolidates memories. The only inputs
  are conversation events.
- **Write path.** After each turn `generate_memories_callback`
  (`after_agent_callback`) calls `callback_context.add_session_to_memory()`,
  sending the session's events to `VertexAiMemoryBankService`, which extracts
  facts under the configured topics. (Alternative:
  `add_events_to_memory(events=...)` to send only a subset for incremental
  processing.)
- **Recall path.** `PreloadMemoryTool` queries Memory Bank at the **start of
  each turn** and injects matching memories into the system instruction — no
  explicit tool call. (Alternative: `LoadMemoryTool` for on-demand recall.)
- **Topics gate persistence.** Only `USER_PERSONAL_INFO` / `USER_PREFERENCES` /
  `EXPLICIT_INSTRUCTIONS` (set in `memory_config.py`) are extracted; anything
  outside those topics is not persisted.
- **Local vs deployed.** Locally ADK uses `InMemoryMemoryService` (nothing
  persists across restarts). Real Memory Bank is used only when
  `memory_service_uri=agentengine://…` — Cloud Run wires this automatically;
  `adk web` needs the flag.
- **Consolidation is asynchronous.** Memories are extracted/consolidated after
  the turn, so a fact told in session 1 shows up in a *later* session, not
  necessarily instantly within the same one.

## Gotchas / things to know

- **Cloud Run still needs an Agent Engine.** Even the "container" path finds or
  creates an Agent Engine instance for session + memory storage and points both
  service URIs at it — **Memory Bank always lives on Agent Engine.**
- **Both deploy paths enable Memory Bank the same way:** wrap `memory_bank_config`
  in `ReasoningEngineContextSpec` → `context_spec`. The two paths differ on
  updates: `deploy.py` (Agent Engine) re-applies `context_spec` via
  `agent_engines.update(...)`, but the **Cloud Run** path (`fast_api_app.py`)
  matches an existing engine by `display_name` and reuses it **as-is** — a
  pre-existing engine there will **not** pick up `memory_bank_config` changes.
- **Local memory is in-memory only.** `make playground` uses
  `InMemoryMemoryService`; memories vanish on restart. To exercise real Memory
  Bank locally:
  `uv run adk web . --port 8501 --memory_service_uri=agentengine://<RESOURCE_NAME>`.
- **`app/__init__.py` defaults `GOOGLE_CLOUD_LOCATION=global`,** but Agent Engine
  / Memory Bank need a **real region** — `deploy.py` and `fast_api_app.py`
  default to `us-west1`, and `deploy-cloud-run` hard-codes `us-west1`. Keep them
  consistent.
- **Model comes from the `MODEL_NAME` env var.** `.env.example` sets
  `gemini-3.5-flash`; the e2e test/stubs default to `gemini-3.5-flash`. Set
  `MODEL_NAME` explicitly (repo policy: use `gemini-3.5-flash`, not the
  deprecated 2.x models).
- **`INTEGRATION_TEST=TRUE` mocks external services** (Cloud Logging, artifacts)
  so tests run credential-less; `test_server_e2e.py` also sets
  `USE_IN_MEMORY_SESSION=true` and patches Gemini via `_e2e_bootstrap`.
- **Only one automated test asserts the memory wiring**
  (`test_agent_has_memory_wired`); every other test mocks the LLM and uses
  in-memory services, so cross-session recall is not covered end-to-end.

## Where to run things

`Makefile` targets:

- `make install` — `uv sync` (installs `uv` first if missing).
- `make playground` — local ADK web UI on port `8501` (uses
  `InMemoryMemoryService`; select the `app` folder).
- `make local-server` — run the Cloud Run entry point (`app.fast_api_app:app`)
  locally with hot-reload via `uvicorn` (`PORT` overridable).
- `make deploy-agent-engine` — deploy to Agent Engine via
  `app/app_utils/deploy.py` (`AGENT_IDENTITY=true`, `SECRETS="KEY=SECRET_ID,…"`
  optional).
- `make deploy-cloud-run` — `gcloud beta run deploy memory-bank-sample` (region
  `us-west1`; `IAP=true`, `PORT=…` optional).
- `make test` — `uv run pytest tests/unit` then `tests/integration`.
- `make lint` — `codespell` + `ruff check` + `ruff format --check` + `ty check`.

## Reuse (copy as-is)

- **`app/app_utils/memory_config.py` is the reusable core.** Self-contained (only
  `vertexai._genai.types`) — copy it to declare which managed/custom topics
  Memory Bank should extract; both deploy paths consume it unchanged.
- **The memory wiring is two lines in any ADK agent:** add `PreloadMemoryTool()`
  to `tools` (recall) and `after_agent_callback=generate_memories_callback`
  (write, via `add_session_to_memory()`). There is **no coupling** to the demo
  `get_weather` / `get_current_time` tools.
- **The deploy entrypoints are drop-in:** `app/agent_engine_app.py` +
  `app/app_utils/deploy.py` (Agent Engine) and `app/fast_api_app.py` (Cloud Run)
  each read `memory_bank_config` and enable Memory Bank via
  `ReasoningEngineContextSpec` / `context_spec` — copy whichever target you need.
- **`app/app_utils/telemetry.py` and `typing.py`** are generic (OpenTelemetry
  setup + Pydantic `Feedback`/`Request` models) and carry no memory logic —
  reuse independently.
- **Dependencies:** `google-adk` plus
  `google-cloud-aiplatform[agent-engines]` (Memory Bank types live under
  `vertexai._genai.types`); `fastapi` / `uvicorn` are needed only for the Cloud
  Run path.
