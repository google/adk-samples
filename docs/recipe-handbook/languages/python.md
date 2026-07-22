<!-- word count: 842 (target 1200, cap 1800) -->

# Python Recipes

Everything Python-specific in one place. Universal requirements
live in [anatomy](../anatomy.md).

## Minimum Python version

`>= 3.11`. CI enforces via `[project].requires-python`.

## `pyproject.toml`

Required. Generate with the `align-recipe-pyproject` AI skill —
it enforces the conventions below and preserves existing
comments.

**Required fields:**

- `[project].name` — must equal the recipe folder basename. A
  recipe at `contrib/python/my-recipe` needs `name = "my-recipe"`.
- `[project].requires-python` — cannot permit versions below
  3.11.
- `[project].description` — if present, must exactly match
  `manifest.description` (after `strip()`).
- `[project].version` — any valid version.
- `[build-system]` — required. Recipes use `hatchling`:

      [build-system]
      requires = ["hatchling"]
      build-backend = "hatchling.build"

**Forbidden:**

- `[tool.ruff]` or `[tool.ruff.*]` — any Ruff config in the
  recipe. Centralised in the repo root `pyproject.toml`.
- Standalone `ruff.toml` or `.ruff.toml` anywhere under the
  recipe.

**Package layout:**

Recipes name their Python package `app` by convention:

    contrib/python/my-recipe/
      pyproject.toml
      app/
        __init__.py
        agent.py
      tests/
        test_runnability.py

The root `pyproject.toml` sets `known-first-party = ["app"]` for
isort, so this naming is what keeps imports sorted correctly
across recipes.

## `uv.lock`

Required. Generate by running `uv lock` from the recipe root:

    cd contrib/python/my-recipe
    uv lock

Regenerate whenever `pyproject.toml` dependencies change.

## `.env.example`

Required. Declares every environment variable the recipe reads
from Python source.

**Common keys** (`python-validate-recipe.yml` emits a notice if
these are missing but doesn't fail):

- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `MODEL_NAME`

**Enforced:** every `os.getenv(...)`, `os.environ[...]`, and
`os.environ.setdefault(...)` call in the recipe's Python source
must have a matching entry in `.env.example`. Missing entries
fail CI. `check_env_vars.py` detects them by AST-parsing your
Python source (Abstract Syntax Tree — it reads the code
structure, not the running program), so it handles multi-line
calls and import aliases like `from os import getenv`.

Generate or update with the
`extract-python-environment-variables` AI skill. It scans the
source, populates `.env.example` with real defaults extracted
from the code, and never overwrites lines you authored.

**Format:**

    # Google Cloud
    GOOGLE_CLOUD_PROJECT=your-project-id
    GOOGLE_CLOUD_LOCATION=us-central1

    # Model
    MODEL_NAME=gemini-3.5-flash

## `load_dotenv()` placement

Call `load_dotenv()` in the package's `__init__.py`, not
`agent.py`:

    # app/__init__.py
    from dotenv import load_dotenv

    load_dotenv()

    from . import agent  # noqa: E402

`agent.py` can read environment variables at import time. If
`load_dotenv()` is also in `agent.py`, those reads happen before
the `.env` file loads. Putting `load_dotenv()` in `__init__.py`
ensures the `.env` file loads first, because `__init__.py` runs
first when the package is imported.

Add `python-dotenv >= 1.0.0` to `pyproject.toml` dependencies.
The `extract-python-environment-variables` skill handles both.

## Model names

Use `gemini-3.5-flash`. Do NOT hardcode any model name in
source. Read from an environment variable:

    model_name = os.getenv("MODEL_NAME")

Replace any existing `gemini-2.0-flash` or `gemini-2.5-flash`
literals (both deprecated) by running
`extract-python-environment-variables` — it finds hardcoded
model literals (`gemini-*`, `claude-*`, `llama-*`, …), rewrites
them to `os.getenv(...)` calls, and adds the entry to
`.env.example`.

`python-validate-recipe.yml` emits a `::notice` for hardcoded
model literals but does not fail.

## Ruff

One Ruff configuration for the whole repo, in the root
`pyproject.toml`:

- Line length 80
- Double quotes, space indent
- Rules: `E`, `F`, `I`, `C`, `PL`, `B`, `UP`, `RUF`

Recipes must not override any of it (see "Forbidden" above).

Run from the repo root:

    uv run ruff format contrib/python/my-recipe
    uv run ruff check contrib/python/my-recipe

## `tests/test_runnability.py`

Required. This test loads `app/agent.py` and asserts
`root_agent is not None` (and `app is not None` if the recipe
defines one). Its only job is to confirm your agent code does
not crash on import — no real API calls are made, and
import-time side effects like `vertexai.init` and
`google.auth.default` are mocked.

Generate with the `generate-python-runnability-test` AI skill.
It AST-parses `agent.py` to figure out which import-time side
effects need mocking and which env vars need setting.

Missing this file fails `python-tests.yml`.

## Integration tests

`python-tests.yml` skips two patterns so the PR check stays
fast and credential-free:

| Pattern | Excluded |
|---|---|
| `tests/integration/` | Everything under this directory |
| `**/test_integration.py` | Files with this exact name, any depth |

Put tests that hit real APIs in one of these. Run locally:

    cd contrib/python/my-recipe
    uv run pytest tests/integration    # integration only
    uv run pytest                      # full suite

## Local commands

From the repo root:

    # All structural checks at once (manifest + structure + README)
    uv run validate contrib/python/my-recipe

    # Individual validators (for isolating one failure)
    uv run validate manifest contrib/python/my-recipe
    uv run validate structure contrib/python/my-recipe
    uv run validate readme contrib/python/my-recipe

    # Format and lint
    uv run ruff format contrib/python/my-recipe
    uv run ruff check contrib/python/my-recipe

    # Tests (mirrors CI — integration excluded)
    cd contrib/python/my-recipe
    uv run pytest --ignore=tests/integration \
      --ignore-glob="**/test_integration.py"

## AI skills for Python recipes

Full catalog: [Python skills, in the skills catalog](../skills-catalog.md#python-skills).
Quick reference:

| Skill | Job |
|---|---|
| `scaffold-python-recipe` | Create a new recipe with compliant layout |
| `generate-manifest` | Populate `manifest.yaml` |
| `align-recipe-pyproject` | Align `pyproject.toml` with standards |
| `extract-python-environment-variables` | Populate `.env.example`, add `load_dotenv()`, replace hardcoded models |
| `generate-python-runnability-test` | Write `tests/test_runnability.py` |
| `prepare-python-recipe` | Runs all of the above plus ruff and `uv lock` |

For the fastest path, run `prepare-python-recipe` end-to-end —
see the
[prepare-python-recipe entry in the skills catalog](../skills-catalog.md#prepare-python-recipe).

---

← [Checklist](../../recipe-checklist.md) · [Handbook](../README.md)
