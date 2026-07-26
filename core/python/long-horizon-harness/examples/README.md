# Examples

Small, self-contained patterns for running and adapting the sample. Each file sets a
few env vars and serves `horizon.fast_api_app:app` **offline** (no GCP credentials,
no extras) — `gemini-3.6-flash` with in-memory services — so you can read it, run
it, and lift the extension point you need. Background: [`../docs/extending.md`](../docs/extending.md), [`../README.md`](../README.md), [`../docs/quickstart.md`](../docs/quickstart.md).

| Example | Demonstrates | Run |
|---|---|---|
| [`minimal_agent.py`](minimal_agent.py) | Smallest harness: Gemini, tools on the host, in-memory everything. | `uv run uvicorn examples.minimal_agent:app --port 8001` |
| [`custom_sandbox_backend.py`](custom_sandbox_backend.py) | Install a custom `BaseEnvironment` via `set_environment_provider(factory)`, with a clearly-marked in-memory stub backend. | `uv run uvicorn examples.custom_sandbox_backend:app --port 8001` |
| [`agent_platform_backend.py`](agent_platform_backend.py) | A managed-platform `BaseEnvironment`: a real `spawn_process` returning a `ProcessHandle` plus per-turn `refresh_auth` for short-lived platform tokens, via `set_environment_provider`. | `uv run uvicorn examples.agent_platform_backend:app --port 8001` |
| [`extra_tools_and_skills.py`](extra_tools_and_skills.py) | Mount your own route with `app.include_router(...)`; where tools (edit `horizon/agent.py`) and workspace skills live. | `uv run uvicorn examples.extra_tools_and_skills:app --port 8001` |

> These examples show how to **run and adapt the sample**. To *learn a pattern and
> lift it into your own ADK agent*, read the real implementation named in
> [`../AGENTS.md`](../AGENTS.md) — that production code is the documentation.

Once running, drive the agent over A2A at `http://127.0.0.1:8001/a2a` or read
`http://127.0.0.1:8001/lha/sessions`.
