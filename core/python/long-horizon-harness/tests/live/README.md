# Live tests — opt-in, hit real Vertex AI

These tests require live GCP credentials and an enabled Vertex AI / Agent
Platform API. They are **excluded from the default `pytest` run** by the
`testpaths` setting in `pyproject.toml` — running `uv run pytest` alone
must stay all-green for the deterministic suite.

## When to run

Use these to smoke-test that the real ADK runtime path still works after
a non-trivial change (model swap, callback wiring, FastAPI server config).
For everything else, prefer the deterministic suite under `tests/unit/`
and `tests/integration/` which uses `InMemoryMemoryService` /
`InMemorySessionService`.

## How to run

```bash
# Ensure GOOGLE_CLOUD_PROJECT and credentials are set:
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=<your project>

# Then run the live tests explicitly:
uv run pytest tests/live -v
```

## Why a separate directory

Per `CLAUDE.md`:
> Unit tests do NOT need GCP — they use `InMemoryMemoryService`/`InMemorySessionService`.
> Only `agents-cli run`, `agents-cli playground`, and `agents-cli eval run` hit Vertex.

These scaffold-generated tests violate that rule by calling `Runner.run()`
against the real `root_agent` or spawning a real `uvicorn` server. Moving
them out of `tests/integration/` keeps the default suite hermetic (an upstream
agents-cli scaffold limitation).
