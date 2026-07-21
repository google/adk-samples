# Recipe Guidelines

Minimum requirements to get a recipe merged, and the skills that satisfy them for
you.

## Minimum requirements

- **Location:** one of
  - `core/<language>/<name>` — e.g. `core/python/rag-agent-search`
  - `contrib/<language>/<name>` — e.g. `contrib/java/hello-agent`
  - `core/<name>` or `contrib/<name>` — flat layout is allowed;
    `manifest.language` then determines the recipe's language.
  - `skills/<name>` — reserved for skill recipes (see the `SKILL.md`
    requirement below).
- **Directory name:** 30 characters max (lowercase + hyphens only).
- **Size & files:**
  - `core/`: Max **500 files / 50 MB**
  - `contrib/`: Max **70 files / 2 MB**

  (Excluding files and folders like `uv.lock`, `__pycache__/`, `node_modules/`, etc.)

- **Required files — every recipe:**

  | File | Must satisfy |
  |------|--------------|
  | `manifest.yaml` | Valid per schema; `ownership.team` and `ownership.poc` set (no placeholders). Includes `language:` — the single source of truth for language-based requirements. |
  | `README.md` | What the recipe does, setup, and run instructions. |

- **Required files — folder-specific:**

  | Folder | Additional required file(s) |
  |--------|-----------------------------|
  | `core/`   | `AGENTS.md` |
  | `contrib/`| _(none)_ |
  | `skills/` | `SKILL.md` |

- **Required files — Python recipes (declared by `language: python`):**

  | File | Must satisfy |
  |------|--------------|
  | `pyproject.toml` | Every Python recipe must have a `pyproject.toml` file. |
  | `uv.lock` | Generated with `uv lock`; in sync with `pyproject.toml`. |
  | `.env.example` | Declares every environment variable the recipe reads (e.g. `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `MODEL_NAME`). |
  | `tests/test_runnability.py` | Imports the agent and asserts `root_agent` is not `None`. |

- Call `load_dotenv()` in the package `__init__.py` (not `agent.py`) to load environment variables.
- **Ruff** formatting and lint pass using the root `pyproject.toml` config
  (no local `[tool.ruff*]` block, no standalone `ruff.toml` / `.ruff.toml`).

> The full, current requirement matrix lives in
> [`.github/policy.yml`](../.github/policy.yml) (`required_files:` block).
> That file is the source of truth CI enforces; the tables above summarise it.

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

## Verify a Recipe Before the PR

From the repository root:

```bash
# Run every validation check (manifest + structure) on every recipe.
uv run validate

# Run every check against one recipe:
uv run validate core/python/my-recipe

# Run only the language-agnostic structural checks (required files,
# folder name, size, manifest schema) — matches what
# validate-recipe-structure.yml enforces in CI.
uv run validate structure core/python/my-recipe

# Run only the manifest schema check.
uv run validate manifest core/python/my-recipe

# Python-only: format and lint (Ruff config lives in the root pyproject.toml).
uv run ruff format core/python/my-recipe && uv run ruff check core/python/my-recipe
```

To run recipe tests (mirroring what CI does — integration tests excluded):

```bash
cd core/python/my-recipe
uv run pytest --ignore=tests/integration --ignore-glob="**/test_integration.py"
```

## Tests

### Runnability test (required)

Every Python recipe **must** contain `tests/test_runnability.py`. This is a
lightweight smoke test that imports the recipe's agent module and asserts that
`root_agent` is not `None` (and `app is not None` if the recipe defines one).
It does not make real API calls — import-time side effects such as
`vertexai.init` and `google.auth.default` are mocked automatically by the
generated test.

Generate or regenerate it with the `generate-python-runnability-test` skill:

```
"generate runnability test for core/python/my-recipe"
```

CI enforces its presence: a recipe without `tests/test_runnability.py` fails
the `python-tests.yml` check.

### Integration tests (skipped in CI)

The `python-tests.yml` workflow excludes integration tests so the required PR
check stays fast and credential-free. Two patterns are always excluded:

| Pattern | What is excluded |
|---------|-----------------|
| `tests/integration/` | The entire directory and everything beneath it. |
| `**/test_integration.py` | Any file with that exact name, at any depth. |

Place any test that makes real API calls (Vertex AI, Cloud Storage, etc.) in
one of those locations. They will not block the PR check but can be run
locally:

```bash
cd core/python/my-recipe

# Run only integration tests
uv run pytest tests/integration

# Run the full suite including integration tests
uv run pytest
```

## What CI runs

Two workflows validate every PR that touches a recipe:

- **`validate-recipe-structure.yml`** — language-agnostic. Runs the same
  checks as `uv run validate structure`: manifest presence & schema,
  folder name, folder size and file count, and the required-files
  matrix from [`.github/policy.yml`](../.github/policy.yml). Also picks
  up `skills/` recipes once that folder lands.
- **`python-validate-recipe.yml`** — Python-specific only. Runs the
  Ruff-config check, the `.env.example` env-var extraction check, the
  `[project]` metadata check, and the hardcoded-model-name notice.
  Structural checks (folder name, size, required files, manifest schema)
  are NOT re-run here — the structure workflow already handles them.
- **`python-tests.yml`** — runs `pytest` for every Python recipe touched
  by the PR (or all recipes on a manual `workflow_dispatch`). Requires
  `tests/test_runnability.py` to exist. Integration tests under
  `tests/integration/` and files named `test_integration.py` are
  automatically excluded.
