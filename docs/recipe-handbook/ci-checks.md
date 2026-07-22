<!-- word count: 552 (target 800, cap 1200) -->

# CI Checks

Every workflow that runs on a recipe PR, and what it enforces.

## Universal (language-agnostic)

### `validate-recipe-structure.yml`

Runs on every PR touching a recipe.

**Enforces:**

- Recipe location matches an allowed pattern (see
  [anatomy.md#where-a-recipe-lives](./anatomy.md#where-a-recipe-lives)).
- Folder name matches `^[a-z][a-z-]*$`, max 30 chars.
- File count and size are under the configured limits.
- `manifest.yaml` present and valid per
  [schema](../../.github/schemas/manifest-schema.json).
- `ownership.team` and `ownership.poc` are not placeholders.
- Required files exist:
  - `README.md` always
  - `pyproject.toml`, `uv.lock`, `.env.example`,
    `tests/test_runnability.py` for `language: python`
- `README.md` content:
  - No `TODO:` placeholders.
  - At least 100 words.
  - A setup section heading (`Setup`, `Prerequisites`,
    `Installation`, `Requirements`, `Configuration`,
    `Getting Started`, `Before You Begin`, `Environment`).
  - A run section heading (`Run`, `Running`, `Usage`,
    `Quickstart`, `Start`, `Deploy`, `Launch`, `How to Run`)
    plus at least one fenced code block.

Failure symptoms and fixes: see
[troubleshooting.md](./troubleshooting.md).

## Python

### `python-validate-recipe.yml`

Runs on every PR touching a Python recipe.

**Fails on:**

- `.env.example` missing entries for env vars used in Python
  source (AST-based, via `check_env_vars.py`).
- `pyproject.toml` contains a `[tool.ruff*]` block.
- Standalone `ruff.toml` or `.ruff.toml` anywhere in the recipe.
- `[project].name` doesn't equal the recipe folder basename.
- `[project].requires-python` permits versions below 3.11.
- `[project].description`, if set, doesn't match
  `manifest.description`.

**Advisory notices (don't fail):**

- Missing common env keys in `.env.example`:
  `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `MODEL_NAME`.
- Hardcoded model names in `.py` files: `gemini-*`, `claude-*`,
  `llama-*`, etc.

### `python-tests.yml`

Runs on every PR touching a Python recipe.

**Enforces:**

- `tests/test_runnability.py` exists.
- All non-integration tests pass.

**Excludes:** `tests/integration/` and `**/test_integration.py`
— they never run in CI.

### `python-format.yml`

**Enforces:**

- `uv run ruff format --check` passes.
- `uv run ruff check` passes.

Ruff configuration lives in the root `pyproject.toml`. Recipes
cannot override it (see `python-validate-recipe.yml`).

To auto-fix:

    uv run ruff format <recipe-path>
    uv run ruff check --fix <recipe-path>

### `python-dependency-policy.yml`

Runs on every PR that changes a `uv.lock` or a `pyproject.toml`.
Six checks:

**Fails on:**

- **Non-PyPI URLs.** Lockfiles must only reference `pypi.org`
  and `files.pythonhosted.org`. Internal registries, GitHub
  Packages, Artifactory, etc. are all rejected.
- **VCS (git) dependencies.** No `source = { git = "..." }` in
  `uv.lock`. Git deps are non-reproducible and bypass registry
  trust.
- **Local path dependencies.** No `path`, `editable`, or
  `directory` sources — they only exist on the committer's
  machine. Exception: the self-referential `editable = "."`
  entry uv writes for the workspace root package itself.
- **Missing package hashes.** Every distribution must have a
  `sha256`. Missing hashes indicate the lockfile was tampered
  with or generated incorrectly.
- **Stale lockfile.** `uv lock --check` must pass — every
  `uv.lock` must be in sync with its sibling `pyproject.toml`.
- **Missing lockfile.** Every `pyproject.toml` with a
  `[project]` table or `[tool.uv]` section must have a sibling
  `uv.lock`.

To fix a stale lockfile:

    cd <recipe-path>
    uv lock

## Java / Go / TypeScript / Kotlin

No language-specific workflows yet.
`validate-recipe-structure.yml` still applies (it's
language-agnostic).

## When a workflow is triggered

- **PR touching a recipe:** the language workflows for that
  recipe's language run.
- **PR touching workflow infrastructure** (workflow files
  themselves, checker scripts, root `pyproject.toml`): the
  affected workflow re-runs against **every** recipe of that
  language, so the new configuration is applied everywhere.
- **`workflow_dispatch`** (manual trigger from the Actions tab):
  runs against every recipe of the target language.

---

← [Checklist](../recipe-checklist.md) · [Handbook](./README.md)
