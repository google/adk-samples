# Recipe Guidelines

Minimum requirements to get a recipe merged, and the skills that satisfy them for
you. Detailed reference lives in [`docs/recipe-guidelines/`](recipe-guidelines/).

## Minimum requirements (Python)

- **Location:** `core/python/<name>` or `contrib/<name>`.
- **Directory name:** matches `^[a-z][a-z-]*$` (lowercase + hyphens, starts with a
  letter, no digits/underscores/uppercase), 30 chars or less.
- **Size & files:** under 1MB (only `uv.lock` is excluded); 50 or fewer files
  total (everything is counted, including `uv.lock`). Setting `large: true` in
  `manifest.yaml` relaxes both to 10MB / 200 files.
- **Model:** use `gemini-3.5-flash`; `gemini-2.0-flash` and `gemini-2.5-flash` are
  deprecated.
- **Required files:**

  | File | Must satisfy |
  |------|--------------|
  | `pyproject.toml` | `[project].name` = folder name; `requires-python` floor `>=3.11`; `description` matches `manifest.description` if set; `[[tool.uv.index]]` declares public PyPI `default = true`; no `[tool.ruff*]` block and no `ruff.toml`. |
  | `uv.lock` | Generated with `uv lock`; in sync with `pyproject.toml`. |
  | `manifest.yaml` | Valid per schema; `ownership.team` and `ownership.poc` set (no placeholders). |
  | `README.md` | What the recipe does, setup, and run instructions. |
  | `.env.example` | Declares `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `MODEL_NAME`, **and every env var the code reads** (undeclared reads fail CI). Values are `<TODO: update-this-value>` placeholders. |
  | `tests/test_runnability.py` | Imports the agent and asserts `root_agent` (and `app`) are not `None`, without real credentials. |

- **`load_dotenv()`** is bootstrapped in the package `__init__.py`, not `agent.py`.
- **Ruff** formatting and lint pass using the root `pyproject.toml` config (line
  length 80, double quotes).

**Non-Python recipes** still require `manifest.yaml` (correct `language`),
`README.md`, and `.env.example`; dependency management and tests follow that
language's conventions.

## Skills that do the work

Ask your AI assistant to run these. Each takes the recipe path (e.g.
`core/python/my-recipe`) and most support a dry-run preview.

| Skill | Use it to |
|-------|-----------|
| `scaffold-python-recipe` | Create a new recipe with the compliant directory layout and template files. |
| `generate-manifest` | Inspect the recipe and generate a populated `manifest.yaml`. |
| `extract-python-environment-variables` | Populate `.env.example`, add the `load_dotenv()` bootstrap, and replace hardcoded model names with `os.getenv(...)`. |
| `align-recipe-pyproject` | Fix `pyproject.toml` to the rules above. |
| `generate-python-runnability-test` | Write `tests/test_runnability.py` with the right import-time mocks. |
| `prepare-python-recipe` | Run all of the above plus ruff and `uv lock`, end to end. |

**Fastest path:** `"Prepare core/python/my-recipe end to end."` runs
`prepare-python-recipe`, which chains every skill and check in order.

## Verify before the PR

```bash
uv run validate core/python/my-recipe   # manifest.yaml
uv run ruff format core/python/my-recipe && uv run ruff check core/python/my-recipe
cd core/python/my-recipe && uv run pytest
```

CI (`validate-python-recipe.yml`, `validate-manifest.yml`, `validate-lockfiles.yml`,
`python-ruff.yml`, `python-tests.yml`) enforces every requirement above on the PR.
