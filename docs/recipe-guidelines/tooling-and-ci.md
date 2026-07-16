# Tooling & CI

← Back to the [Recipe Guidelines hub](README.md)

---

## 🛠️ Developer Tools, Validation & Formatting

To streamline local development, validate coding standards, and automate
bootstrapping, we provide helper scripts in `tools/`, integration with
**Ruff** for formatting/linting, and agent skills under `.agents/skills/`.

## Code Quality & Formatting (Ruff)

Python code must be formatted and linted with **Ruff** (rules defined in the root [pyproject.toml](../../pyproject.toml)). Always run formatting commands from the **repository root** targeting your specific recipe folder:

*   **Format:**
    ```bash
    uv run ruff format core/<recipe-name>
    ```
*   **Lint & Fix:**
    ```bash
    uv run ruff check --fix contrib/<recipe-name>
    ```

---

## Local Validation Suite (`tools/`)

Run the validator locally to catch `manifest.yaml` issues before CI.

### Setup
Before running the validator for the first time, run this command from the **repository root**:
```bash
uv sync
```

### Usage
Run the validation tool using the following syntax:
```bash
uv run validate [subcommand] [scope]
```

*   **`subcommand`** (Optional): `manifest` (validates `manifest.yaml` schema/placeholders) or `all` (default; runs every registered check — **currently only `manifest`**).
*   **`scope`** (Optional): `all` (default), `core`, `contrib`, or specific directory path (e.g., `core/rag-agent-search`).

### Examples
*   **Validate everything:**
    ```bash
    uv run validate
    ```
*   **Run only manifest validation on all recipes:**
    ```bash
    uv run validate manifest
    ```
*   **Run all checks on a single recipe:**
    ```bash
    uv run validate core/rag-agent-search
    ```

> **Note:** `uv run validate` currently checks only `manifest.yaml`. Directory
> size, naming, required files, `.env.example` variables, Ruff formatting, and
> runnability are enforced separately in CI — see [Continuous Integration](#continuous-integration)
> below. Run Ruff and `uv run pytest` locally before opening a PR.

---

## Developer Agent Skills (`.agents/skills/`)

AI coding assistants (like Antigravity or Gemini) can use standard instructions under `.agents/skills/` to automate setup tasks.

### Available Skills

#### `scaffold-python-recipe`
*   **Purpose**: Creates a compliant directory structure with the template files (`app/`, `tests/`, `pyproject.toml`, `README.md`, `.env.example`, `manifest.yaml`) by running the scaffold script automatically. Run `uv lock` afterwards to generate `uv.lock`.
*   **Examples to trigger this skill**:
    *   > *"Help me create a new recipe named `weather-alert-agent`"*
    *   > *"Scaffold a Python recipe for a database-query assistant inside `contrib/`"*

#### `generate-manifest`
*   **Purpose**: Inspects your recipe's source code, structure, and configuration to generate a correct, fully populated `manifest.yaml`, then runs local validation to confirm it passes.
*   **Examples to trigger this skill**:
    *   > *"Generate a manifest.yaml for the recipe `contrib/weather-alert-agent`"*
    *   > *"Inspect the code under `core/search-assistant` and write its manifest"*

#### `extract-python-environment-variables`
*   **Purpose**: Scans all non-test Python files in a recipe for environment variable reads (`os.getenv`, `os.environ.get`, `os.environ[]`), then automatically:
    *   Creates or updates `.env.example` with any missing variables. **Every entry is written with `<TODO: update-this-value>` as its value** — the skill never persists inferred defaults from source code, even when the code has an `os.getenv("VAR", "some_default")` fallback.
    *   Detects hardcoded model name strings and replaces them with **bare `os.getenv("MODEL_NAME")` calls** (no default argument — the skill never writes defaults into Python source either). When a recipe uses multiple models, each gets a distinct name derived from the model string (e.g. `MODEL_NAME_GEMINI_3_5_FLASH`, `MODEL_NAME_TEXT_EMBEDDING_004`), with a numeric suffix (`_2`, `_3`, …) added only if two models would otherwise collide.
    *   Injects the standard `load_dotenv()` bootstrap snippet into the recipe's package `__init__.py` (e.g. `app/__init__.py`).
    *   Ensures `python-dotenv>=1.0.0` is listed in `pyproject.toml` dependencies.
*   **Hard rules the skill NEVER breaks**: (1) no inferred defaults anywhere — `.env.example` gets placeholders only; (2) additive-only for Python files — never writes new `os.environ.setdefault(...)` bootstrap lines, and pre-existing `os.environ.setdefault(...)` or `os.getenv("VAR", "default")` calls the recipe author wrote by hand are left untouched.
*   **Preview mode**: Pass `--dry-run` (or ask to "preview") to report everything that *would* change without modifying any files.
*   **Examples to trigger this skill**:
    *   > *"Extract environment variables for `contrib/weather-alert-agent`"*
    *   > *"Preview the environment variable changes for `core/python/rag-agent-search` (dry run)"*
    *   > *"Fix the environment variable setup for my recipe"*

#### `align-recipe-pyproject`
*   **Purpose**: Auto-aligns a recipe's `pyproject.toml` with the repo standards enforced by CI. Runs six checks; most are auto-fixable via comment-preserving TOML edits (`tomlkit`):
    *   **`no-local-ruff-config`** — removes any `[tool.ruff*]` table (ruff config is centralized in the root `pyproject.toml`).
    *   **`python-version-floor`** — rewrites `[project].requires-python` so the lower bound is `>=3.11` (per `AGENTS.md`), preserving any upper bound. Interpretation A: higher floors like `>=3.12` are the author's choice and are left alone.
    *   **`project-name-matches-folder`** — sets `[project].name` to the recipe folder basename.
    *   **`description-matches-manifest`** — if `[project].description` is set, verifies it equals `manifest.description`; resolvable with `--description-source={pyproject,manifest,delete}`.
    *   **`build-system-present`** — report-only (backend choice is editorial; the skill won't pick between hatchling and uv_build for you).
    *   **`default-pypi-index`** — adds `[[tool.uv.index]] url="https://pypi.org/simple/" default=true` if missing. Required so `uv sync` works on Google corp workstations without corp Airlock auth.
*   **Preview mode**: `--dry-run` reports what would change without modifying anything.
*   **Examples to trigger this skill**:
    *   > *"Align pyproject.toml for `core/python/my-recipe`"*
    *   > *"Check what needs fixing in my recipe's pyproject.toml (dry run)"*

#### `generate-python-runnability-test`
*   **Purpose**: Generates `tests/test_runnability.py` for a Python recipe. Parses the recipe's `agent.py` **and** its package `__init__.py` with `ast` to detect what side effects fire at import time, then emits the minimal test needed to import the package cleanly:
    *   Detects top-level `root_agent` and `app` assignments (so the test asserts on the right names).
    *   Detects `vertexai.init(...)` calls or `import vertexai` → wraps the import in `with patch("vertexai.init"):`.
    *   Detects `google.auth.default()` calls or `import google.auth` → sets `GOOGLE_CLOUD_PROJECT=test-project` AND patches `google.auth.default` to return a fake `(credentials, project_id)` tuple (needed for recipes that call it unconditionally at import time).
    *   Detects `INTEGRATION_TEST` env-var reads anywhere in the recipe → sets `INTEGRATION_TEST=TRUE` before the import.
    *   Emits a minimal test (no boilerplate) when no side effects are detected.
*   **Overwrite**: Refuses to clobber an existing `tests/test_runnability.py` unless `--overwrite` is passed.
*   **Examples to trigger this skill**:
    *   > *"Create a runnability test for `core/python/my-recipe`"*
    *   > *"Generate tests/test_runnability.py for this recipe"*

#### `prepare-python-recipe` (master orchestrator)
*   **Purpose**: End-to-end orchestration that runs the four sub-skills above (plus ruff + `uv lock` + `py_compile`) in the right order on an already-in-place recipe. Seven phases:
    1.  Generate `manifest.yaml` (if missing; then verify team + POC).
    2.  Extract env vars (`extract-python-environment-variables`).
    3.  Align `pyproject.toml` (`align-recipe-pyproject`).
    4.  Ruff `format` + `check --fix` (runs **after** align, because align removes any local `[tool.ruff*]` block that would otherwise shadow the root config).
    5.  Recipe `uv lock` (regenerates `uv.lock` against the aligned `pyproject.toml`; does NOT install into `.venv/` — that's a heavier step the user runs after reviewing).
    6.  Generate `tests/test_runnability.py` (`generate-python-runnability-test`).
    7.  `py_compile` the generated test file (lightweight sanity check that it's valid Python; does NOT execute the test).
*   **Interactive by design**: pauses at fixed checkpoints (manifest team/POC verification, description mismatch, existing test regeneration) and at judgment interruptions (e.g. suspicious dependencies, unexpected detections). Never commits.
*   **Prerequisites** the user must complete manually before invoking: (a) deactivate any active venv; (b) `git pull` and `uv sync` at the repo root; (c) recipe placed at its target path; (d) commit the original recipe first, so `git diff` shows what the pipeline changed.
*   **Examples to trigger this skill**:
    *   > *"Prepare `core/python/my-recipe` end to end"*
    *   > *"Run all the checks and fixes on this recipe"*
    *   > *"Make this recipe PR-ready"*

---

## Continuous Integration

Pull requests are gated by GitHub Actions workflows in
[.github/workflows/](../../.github/workflows/). The checks most relevant to recipe
contributors are:

| Workflow | What it enforces |
|----------|------------------|
| `python-validate-recipe.yml` | Directory size & file-count limits (tunable — see [`.github/policy.yml`](../../.github/policy.yml)), directory-name rules, required files, `.env.example` completeness (every variable used in code must be declared), and the six `pyproject.toml` rules from the `align-recipe-pyproject` skill: no `[tool.ruff*]` block, no standalone `ruff.toml` / `.ruff.toml` files, `[project].name` matches the folder basename, `[project].requires-python` has a `>=3.11` (or higher) floor, `[project].description` matches `manifest.description` if set, and `[[tool.uv.index]]` declares public PyPI as `default = true`. The `pyproject.toml` checks are delegated to `.github/scripts/check_recipe_pyproject.py`. |
| `validate-manifest.yml` | `manifest.yaml` schema and placeholder values (same as `uv run validate`). |
| `python-dependency-policy.yml` | `uv.lock` supply-chain and reproducibility policy: only public PyPI is referenced (no internal or `pkg.dev` registries), no VCS (`git = …`) dependencies, no local `path` / `editable` / `directory` dependencies, every distribution has a hash, every lockfile is in sync with its sibling `pyproject.toml`, and every installable `pyproject.toml` has a sibling `uv.lock`. Hash-presence logic is delegated to `.github/scripts/check_lockfile_hashes.py`. |
| `python-ruff.yml` | Ruff formatting & lint on changed Python files. Uses the root `pyproject.toml`'s `[tool.ruff]` config directly (no CLI overrides in the workflow) — combined with the `no-local-ruff-config` rule above, this makes the root `pyproject.toml` the single source of truth for ruff configuration. |
| `python-tests.yml` | Per-recipe test suites (`uv run pytest`) for changed recipes. |

Because `uv run validate` only covers `manifest.yaml`, run Ruff
(`uv run ruff check` / `uv run ruff format`) and your recipe's tests
(`uv run pytest`) locally before opening a PR so your changes match CI.
The `align-recipe-pyproject` and `prepare-python-recipe` skills above
auto-fix most of what `python-validate-recipe.yml` checks — running one
of them on your recipe before pushing is the fastest way to green CI.

---

← Back to the [Recipe Guidelines hub](README.md)
