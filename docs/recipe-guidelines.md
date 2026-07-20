# Recipe Guidelines

Minimum requirements to get a recipe merged, and the skills that satisfy them for
you.

## Minimum requirements

- **Location:** `core/<language>/<name>` or `contrib/<language>/<name>` (e.g.
  `core/python/rag-agent-search`, `contrib/java/hello-agent`).
- **Directory name:** 30 characters max (lowercase + hyphens only)
- **Size & files:**
  - `core/`: Max **500 files / 50 MB**
  - `contrib/`: Max **70 files / 2 MB**

  (Excluding files and folders like `uv.lock`, `__pycache__/`, `node_modules/`, etc.)

- **Required files (All languages):**

  | File | Must satisfy |
  |------|--------------|
  | `manifest.yaml` | Valid per schema; `ownership.team` and `ownership.poc` set (no placeholders). |
  | `README.md` | What the recipe does, setup, and run instructions. |
  | `.env.example` | Declares environment variables like `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `MODEL_NAME` |

- **Required files (Python specific):**

  | File | Must satisfy |
  |------|--------------|
  | `pyproject.toml` | Every python recipe must have a `pyproject.toml` file. |
  | `uv.lock` | Generated with `uv lock`; in sync with `pyproject.toml`. |
  | `tests/test_runnability.py` | Imports the agent and asserts `root_agent` is not `None`. |

- Call `load_dotenv()` in the package `__init__.py` (not `agent.py`) to load environment variables.
- **Ruff** formatting and lint pass using the root `pyproject.toml` config.

## Skills that do the work

Ask your AI assistant to run these. Each takes the recipe path (e.g.
`core/python/my-recipe`) and most support a dry-run preview.

| Skill | Use it to | Example Prompt |
|-------|-----------|----------------|
| `scaffold-python-recipe` | Create a new recipe with the compliant directory layout and template files. | `"scaffold a new Python sample at core/python/my-recipe"` |
| `generate-manifest` | Inspect the recipe and generate a populated `manifest.yaml`. | `"generate manifest.yaml for core/python/my-recipe"` |
| `extract-python-environment-variables` | Populate `.env.example`, add the `load_dotenv()` call, and replace hardcoded model names with `os.getenv(...)`. | `"extract env vars for core/python/my-recipe"` |
| `align-recipe-pyproject` | Align `pyproject.toml` with repository standards. | `"align pyproject.toml for core/python/my-recipe"` |
| `generate-python-runnability-test` | Write `tests/test_runnability.py` with the right import-time mocks. | `"generate runnability test for core/python/my-recipe"` |
| `prepare-python-recipe` | Run all of the above plus ruff and `uv lock`, end to end. | `"prepare core/python/my-recipe end to end"` |

**Fastest path:** `"Prepare core/python/my-recipe end to end."` runs
`prepare-python-recipe`, which chains every skill and check in order.

## Verify a Python Recipe Before the PR

From the repository root:

```bash
uv run validate manifest core/python/my-recipe
uv run ruff format core/python/my-recipe && uv run ruff check core/python/my-recipe
```

To run tests:

```bash
cd core/python/my-recipe && uv run pytest
```
