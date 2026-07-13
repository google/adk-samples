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
    *   Creates or updates `.env.example` with any missing variables (using inline defaults from the code where available, otherwise `<TODO: update-this-value>`).
    *   Detects hardcoded model name strings and replaces them with `os.getenv("MODEL_NAME")`. When a recipe uses multiple models, each gets a distinct name derived from the model string (e.g. `MODEL_NAME_GEMINI_2_5_FLASH`, `MODEL_NAME_TEXT_EMBEDDING_004`), with a numeric suffix (`_2`, `_3`, …) added only if two models would otherwise collide.
    *   Injects the standard `load_dotenv()` bootstrap snippet into the recipe's package `__init__.py` (e.g. `app/__init__.py`).
    *   Ensures `python-dotenv>=1.0.0` is listed in `pyproject.toml` dependencies.
*   **Preview mode**: Pass `--dry-run` (or ask to "preview") to report everything that *would* change without modifying any files.
*   **Examples to trigger this skill**:
    *   > *"Extract environment variables for `contrib/weather-alert-agent`"*
    *   > *"Preview the environment variable changes for `core/python/rag-agent-search` (dry run)"*
    *   > *"Fix the environment variable setup for my recipe"*

---

## Continuous Integration

Pull requests are gated by GitHub Actions workflows in
[.github/workflows/](../../.github/workflows/). The checks most relevant to recipe
contributors are:

| Workflow | What it enforces |
|----------|------------------|
| `validate-python-recipe.yml` | Directory size & file-count limits, directory-name rules, required files, and `.env.example` completeness (every variable used in code must be declared). |
| `validate-manifest.yml` | `manifest.yaml` schema and placeholder values (same as `uv run validate`). |
| `validate-lockfiles.yml` | `uv.lock` is present and in sync with `pyproject.toml`. |
| `python-ruff.yml` | Ruff formatting & lint on changed Python files. |
| `python-tests.yml` | Per-recipe test suites (`uv run pytest`) for changed recipes. |

Because `uv run validate` only covers `manifest.yaml`, run Ruff
(`uv run ruff check` / `uv run ruff format`) and your recipe's tests
(`uv run pytest`) locally before opening a PR so your changes match CI.

---

← Back to the [Recipe Guidelines hub](README.md)
