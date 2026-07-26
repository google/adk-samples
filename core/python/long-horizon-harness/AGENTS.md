# Studying Long Horizon — a coding-agent's guide

You are a coding agent studying **Long Horizon** as a *teaching sample*: read it to
lift one pattern into your own ADK agent. (Secondarily, you can run/adapt it as a
starter.) This file is your entry point — scan it, jump to the one pattern you need,
read the **real function** that implements it, then go deep.

> **The code is the documentation.** Each pattern below points at the actual
> production function that implements it — not a parallel toy example that would
> drift. Open the named file, read how *we* did it, lift it into your own agent.

> Maintainer-facing internals (exhaustive callback ordering, state keys, every env
> var) live in [`AGENTS.md`](AGENTS.md). Read it last, when you need exact wiring.

## What this sample teaches

Horizon leans on ADK + Vertex primitives; the **six interfaces** below are where custom code
genuinely earns its keep — a real Protocol, ContextVar, or ordered callback chain, the
densest and most liftable, so start here. Each is self-contained:

1. **Environment interface** — tools call a pluggable `Environment`, not the host.
2. **Tool guardrails + exfil/egress** — block/ask on risky tool calls.
3. **Per-user secrets** — act with the user's own credentials, unseen by the model.
4. **Sub-agent delegation + HITL resurfacing** — blocking `delegate` + fire-and-forget `agent`; a child bubbles a risky-op approval to the human and resumes.
5. **Self-improvement loop** — write facts/skills to memory between turns.
6. **3-tier system prompt** — stable cached prefix + per-turn volatile tail.

These are *interfaces*, not knobs: things like **compaction**, **resumability**, and the
**prefix cache** are mostly ADK/Vertex config Horizon only tunes (compaction ships a summarization prompt + banner over
ADK's `LlmEventSummarizer`), and features like **routines** and the **scheduler** are
*applications* composed from the interfaces. Those — plus other subsystems — are curated in
[**Beyond the six**](#beyond-the-six--other-subsystems-worth-studying), and
[`docs/architecture.md`](docs/architecture.md)'s Backend tree map is the complete
per-subsystem index.

## Recipe table — where to study + how to lift

The **Start-here** column names the exact real function to read — open it; that
production code is the lesson.

| Interface | Start-here (real function) | Transferable vs Horizon-specific | Deep-dive |
|---|---|---|---|
| Environment interface | `horizon/environment/base.py` → `Environment`; `horizon/environment_context.py` → `active_environment()` / `set_environment_provider()` | Copy: an `Environment` (Horizon's contract — a superset of ADK's `BaseEnvironment` adding `list_directory`/`delete_file`/`make_dir`/`download_zip`/`upload_zip`/`spawn_process` (returns a `ProcessHandle` from `horizon/environment/process.py`) + capability flags `on_host_fs` + a per-turn `refresh_auth()`) behind a ContextVar; tools call `active_environment()` and dispatch by method/capability, never `isinstance` or the host (zero concrete-class `isinstance` remain). Per-turn auth is **env-owned** (`refresh_auth`), so the light `set_environment_provider` hook alone suffices for a backend with short-lived tokens. Full provisioning/reattach/snapshot/upgrade lives behind a second interface — `SandboxProvider` (`horizon/sandbox/provider.py`), `VertexSandboxProvider` | `LocalProvider`, overridable via `set_sandbox_provider`. Specific: the Agent Runtime REST backend (`horizon/environment/sandbox.py`) + its provisioning subsystem (`horizon/sandbox/`); also the per-session focus lens (`horizon/workspace_window.py` → `resolve_in_window`) — a default/lens, not a security boundary. | [`docs/sandbox-lifecycle.md`](docs/sandbox-lifecycle.md) |
| Tool guardrails | `horizon/guardrails/__init__.py` (package docstring = the **contract**) → `exfil_guard()` (worked example) | Copy the contract the package docstring states: a `before_tool` callback `(*, tool, args, tool_context)` returning `None` to allow or a dict-with-`error` to block, added to `horizon/agent.py`'s `before_tool_callback` list. Specific (skip when lifting): `exfil_guard`'s ~570 lines are exfil-detection heuristics; the three-layer exfil/policy/permission chain (`guardrails_plugin.py` is the sibling halt plugin). Layer D — the interactive ask-layer that runs last (argv classifier so a benign segment can't smuggle a gated one) — is `horizon/guardrails/permission_guard.py` → `permission_guard`. | [`docs/security-model.md`](docs/security-model.md), [`docs/permission-model.md`](docs/permission-model.md) |
| Per-user secrets | `horizon/secrets/store.py` → `SecretStore` Protocol (`SecretManagerStore` \| `InMemorySecretStore`, selected by `LHA_SECRET_BACKEND`, overridable via `set_secret_store`); `horizon/secrets/inject.py` → `secret_env` / `set_routine_secret_scope`; OAuth `horizon/auth/oauth.py` → `attach_gcp_oauth_routes` / `sign_state` / `verify_state` | Copy: a per-user `SecretStore` behind a Protocol + env selector, resolved and scoped behind a single env-injection interface so the agent acts with the user's credentials while the model sees only the name; access-token-only OAuth with an HMAC-signed `state` cross-checked against IAP identity (no refresh token stored). Specific: Secret Manager + the vendor `NotFound`/`AlreadyExists` translation (a fake needs zero GCP imports) + the Connect-Google surface. | [`docs/security-model.md`](docs/security-model.md) |
| Sub-agent delegation + HITL resurfacing | `horizon/subagents/delegate.py` → `delegate()` (fire-and-forget `agent` → `subagents/spawn.py`; resumable child driver `subagents/delegate_runner.py` → `drive_child`) | Copy: blocking `delegate` + fire-and-forget `agent` as root tools, each with its own isolated context window; the delegate drives a resumable child that pauses on a risky-op approval, bubbles it to the human, and resumes from the stored `FunctionResponse` — durable HITL without re-running the turn. Specific: the resurfacing bubble budget + `ask_parent` escalation. | — |
| Self-improvement loop | `horizon/memory/auto_capture.py` → `auto_capture_callback()` | Copy: an `after_agent` callback calling `callback_context.add_session_to_memory()` (an ADK `CallbackContext` primitive) for write-back, plus a `PreloadMemoryTool` in the root agent's `tools` for prefetch = cross-session recall (both wired in `horizon/agent.py`). Backend-specific memory access (profiles + list-all) is confined to one interface — `horizon/memory/adapter.py` (`MemoryAdapter` Protocol + `memory_adapter()` factory), so callers name no concrete ADK service class and a non-Vertex backend degrades cleanly. Specific: dream-review, judge fork, and the skills system — auto-discovery from `SKILL.md` (`horizon/tools/skill_reload.py` → `bind_session_skills_callback`) plus the promote/demote curator (`horizon/memory/skill_curator.py` → `skill_curator_callback`). | [`docs/memory.md`](docs/memory.md) |
| 3-tier system prompt | `horizon/conversation/system_prompt.py` → `make_system_prompt_callback()` / `build_stable_tier()` (volatile tail in `conversation/reminders.py`) | Copy: stable cached prefix + volatile tail injected as trailing system reminders so the cache prefix stays byte-stable. Specific: the soul/skill tiers. | — |

The interfaces above are taught by Horizon's **real code** — nothing to run, nothing
that drifts. Separately, the `examples/` dir shows how to **run and adapt the
sample**: [`examples/minimal_agent.py`](examples/minimal_agent.py)
(smallest harness), [`examples/custom_sandbox_backend.py`](examples/custom_sandbox_backend.py)
(the `Environment` interface via `set_environment_provider`),
[`examples/agent_platform_backend.py`](examples/agent_platform_backend.py) (a
managed-platform backend: `spawn_process` + per-turn `refresh_auth` via the same
light hook), and
[`examples/extra_tools_and_skills.py`](examples/extra_tools_and_skills.py) (add a
route / tool / skill) — for when you want to *run* Horizon, not just lift a pattern.

## Beyond the six — other subsystems worth studying

The six interfaces are where Horizon writes the *most* custom code — not all of it. Some entries
here are ADK/Vertex **knobs** Horizon only tunes (compaction, resumability); some are
**applications** that compose the interfaces (routines, scheduler); the rest are smaller or more
specialized subsystems. Each still carries a production lesson worth lifting. A curated
shortlist; for the complete per-subsystem index see
[`docs/architecture.md`](docs/architecture.md)'s [Backend tree map](docs/architecture.md#backend-tree-map--where-to-start).

| Topic | Start-here (file → symbol) | Why worth studying | Deep-dive |
|---|---|---|---|
| A2A + Gemini Enterprise interop | `horizon/a2a/executor.py` → `_StreamDedupConverter` (+ `_surface_artifact_links`, `_strip_fake_artifact_links`); transport `horizon/a2a/routes.py` → `attach_a2a_routes` | One A2A converter satisfies two non-conformant clients (Gemini Enterprise vs. the web) at once — dedupe streamed text, reshape tool chips, lift artifact links GE buries, strip model-fabricated links. | — |
| Model routing interface | `horizon/models/dispatcher.py` → `DispatchingLlm`; `horizon/models/registry.py` → `_MODELS` (`ModelDescriptor`) + `model_capabilities`; `horizon/models/capabilities.py` → `ModelCapabilities` | Copy: one `BaseLlm` holds every registered backend + a per-backend capability descriptor (media limits + an optional `prepare_contents` content hook) instead of `isinstance`/name-gating. Ships Gemini-only (`gemini-3.6-flash` default + `gemini-3.1-pro`, both via `/model`); adding a model or provider (e.g. a `LiteLlm` entry) is one `_MODELS` table entry. | — |
| Resumability (an ADK knob) | `horizon/agent.py` → `ResumabilityConfig(is_resumable=True)` (one line; durable persistence is Agent Runtime) | Config, not an interface: it's the ADK/Vertex primitive the Sub-agent-delegation interface's child driver (`delegate_runner.py` → `drive_child`) builds on to pause on a risky-op approval and resume from the stored `FunctionResponse`. The custom value is that child driver (interface 4), not resumability itself. | — |
| Routines (unattended cron, isolated sandbox) | `horizon/routines/tools.py` → `_create`; fire path `horizon/scheduler/routine_tick_endpoint.py` → `routine_tick` / `_fire_routine`; isolation `horizon/routines/run_context.py` | A recurring task runs headless in a fresh, disjoint `lhart-` sandbox scoped to only its declared secrets, with non-shell approvals auto-denied — the blast-radius design is the lesson. | [`docs/routines.md`](docs/routines.md) |
| Scheduler (reminders as persisted chats / dream-review / snapshot) | `horizon/scheduler/sessions.py` → `create_scheduled_session`; fire `horizon/scheduler/tick_endpoint.py` → `tick` / `_fire_one`; `horizon/scheduler/dream_review_endpoint.py` → `dream_review_tick`; `horizon/scheduler/snapshot_endpoint.py` → `snapshot_tick` | A scheduled turn drives the *same* shared A2A handler against a pre-tagged session so it records a real Task/history — driving the runner directly would leave the UI blank. | — |
| Context compression (an ADK knob) | `horizon/context/summarizer.py` → `HorizonSummarizer` (subclasses ADK's `LlmEventSummarizer`, an `EventsCompactionConfig` hook) | Mostly config, not an interface: ADK owns the trigger/retention/lifecycle; Horizon supplies only a structured summarization prompt + a REFERENCE-ONLY banner. Its one custom idea — a pre-compaction memory fork (`spawn_flush_fork`) so durable facts land before a lossy summary — is really the self-improvement loop. | [`docs/memory.md`](docs/memory.md) |
| DB connection resilience | `horizon/infrastructure/db_resilience.py` → `retry_on_disconnect` / `resilient_engine_kwargs` / `is_transient_disconnect` | Why `pool_pre_ping` is not enough (it validates only at checkout) and how every op wraps a transient-only retry to survive a Cloud SQL failover mid-query. | — |
| Sandbox lifecycle | `horizon/sandbox/lifecycle.py` → `find_latest_user_sandbox` (version-agnostic reattach) / `snapshot_and_prune_user` / `restore_sandbox_from_snapshot` | Version-scoped identity but version-agnostic reattach (a rollout never wipes installed CLIs); snapshot/restore for TTL survival — the lifecycle math is the hard part. | [`docs/sandbox-lifecycle.md`](docs/sandbox-lifecycle.md) |
| FastAPI serving surface | `horizon/fast_api_app.py` → `_build_app` (mounts A2A + all `/lha/*` + `/scheduler/*`) / `build_runner` (Runner, no FastAPI) | Env-driven serving over the agent; ship a subset by deleting `attach_*` calls. What each route exposes is the security story. | [`docs/security-model.md`](docs/security-model.md) |

Honorable mention — **artifact signed URLs + model redaction**: `horizon/tools/_artifact_links.py` → `artifact_url` / `_signed_blob_url` (a V4 signed URL via IAM SignBlob, no private key) paired with `horizon/context/artifact_url_redaction.py` → `redact_artifact_urls_callback` (the model reads a placeholder, never the credentialed blob).

Still not exhaustive: [`docs/architecture.md`](docs/architecture.md)'s [Backend tree map](docs/architecture.md#backend-tree-map--where-to-start) lists every subsystem with its own start-here file — read this section for the highlights, that map for full coverage.

## Study order

1. Skim [`docs/architecture.md`](docs/architecture.md) — the map and the construction/execution flow.
2. Pick **one** pattern from the table above.
3. Open its **real function** (the Start-here column); fan out to the supporting files the architecture map lists.
4. Read its **deep-dive** doc.
5. Only then consult [`AGENTS.md`](AGENTS.md) for exhaustive wiring (callback order, state keys).
6. To run/adapt the sample in your own app, read [`docs/extending.md`](docs/extending.md) and the `examples/`.

## What to ignore when studying

The chat UI in `web/` (a Vite SPA + Express proxy behind IAP; see `web/README.md`) is a
real part of the repo — read it if the frontend is what you're after; it's just outside the
ADK harness interfaces this guide teaches. When studying the *interfaces*, you can skip:

- `terraform/` — deploy infra; relevant only if you're running Horizon as a starter.
- `tests/eval/` — LLM-behavior validation, not pattern source. (`tests/unit` + `tests/integration` show contracts.)
- Generated/large files: `uv.lock`, `*.db`, `.venv/`, `node_modules/`.

## agents-cli sample metadata

- **name:** `horizon`
- **one-liner:** Self-improving, long-horizon ADK agent on Google Agent Platform — per-user sandbox, cross-session memory, tool guardrails, and a between-turns self-improvement loop.
- **keywords:** long-horizon, self-improving, memory bank, sandbox, guardrails, exfil, egress, sub-agents, delegation, routines, scheduler, a2a, resumable, skills, vertex, agent runtime, oauth, secrets, model-routing, db-resilience
- **key files:** `horizon/agent.py`, `horizon/fast_api_app.py`, `examples/minimal_agent.py`, `docs/architecture.md`, `AGENTS.md`
