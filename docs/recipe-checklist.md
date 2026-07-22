<!-- word count: 830 (target 700, cap 1000) -->

# Recipe Checklist

Everything you need to ship a `contrib/` recipe, on one page.
Deep detail lives in [`recipe-handbook/`](./recipe-handbook/README.md).

---

## AI skills reference

If you use an AI assistant (e.g.
[Antigravity CLI](https://antigravity.google/product/antigravity-cli)),
the following skills automate most of the checklist. The fastest
path is to invoke `prepare-python-recipe` — it runs every other
skill in the right order.

| Skill | What it does | Example prompt |
|---|---|---|
| `prepare-python-recipe` | Runs every other skill below in sequence. The fastest path to a PR-ready recipe. | `prepare the python recipe contrib/python/my-recipe` |
| `generate-manifest` | Writes a valid `manifest.yaml` from your recipe files | `generate manifest for contrib/python/my-recipe` |
| `align-recipe-pyproject` | Fixes `pyproject.toml` to match repo conventions | `align pyproject.toml for contrib/python/my-recipe` |
| `extract-python-environment-variables` | Populates `.env.example` from Python source and adds `load_dotenv()` where needed | `extract env vars for contrib/python/my-recipe` |
| `generate-python-runnability-test` | Writes `tests/test_runnability.py` | `generate runnability test for contrib/python/my-recipe` |

For deep detail on each skill, see the
[Skills Catalog](./recipe-handbook/skills-catalog.md).

If you're not using an AI assistant, § 3 below shows the manual
command sequence.

---

## 1. Always

- [ ] Recipe lives at `contrib/<lang>/<name>` —
      [anatomy](./recipe-handbook/anatomy.md)
- [ ] Recipe name (folder name) ≤ 30 chars, lowercase + hyphens
      only
- [ ] Under size limit: 70 files / 2 MB for `contrib/`
- [ ] `manifest.yaml` valid, with real `ownership.team` and
      `ownership.poc` — AI skill: `generate-manifest`
- [ ] `README.md` has ≥ 100 words, a setup section, and a run
      section with a code block — CI enforces this;
      [details](./recipe-handbook/anatomy.md#readmemd)

## 2. Your language

**Python** — [details](./recipe-handbook/languages/python.md)
- [ ] `pyproject.toml` aligned — AI skill:
      `align-recipe-pyproject`
- [ ] `uv.lock` in sync — run `uv lock` from the recipe root
- [ ] `.env.example` declares every env var the recipe reads —
      AI skill: `extract-python-environment-variables`
- [ ] `load_dotenv()` called in the package `__init__.py` (not
      `agent.py`)
- [ ] Models: use up-to-date models like `gemini-3.5-flash` (not `gemini-2.0-flash` or
      `gemini-2.5-flash`)
- [ ] `tests/test_runnability.py` present — AI skill:
      `generate-python-runnability-test`

**Java / Go / TypeScript / Kotlin** — language-specific guidance
coming soon. Structural checks apply already.

## 3. Before you open the PR

The commands in this section run from the repo root and mirror
what CI runs. They all use `$RECIPE_PATH` — set that variable
first.

### Set your recipe path

Do this once before running any of the commands below:

    export RECIPE_PATH=contrib/python/my-recipe

Replace `my-recipe` with your recipe's folder name. Every
command below assumes this variable is set.

### Structural checks

`uv run validate <recipe-path>` runs all structural validators
against your recipe and reports PASS / FAIL for each.

- [ ] All checks pass:
      ```
      uv run validate $RECIPE_PATH
      ```

Individual validators (useful for isolating one failure):

    uv run validate manifest $RECIPE_PATH
    uv run validate structure $RECIPE_PATH
    uv run validate readme $RECIPE_PATH

- `validate manifest` — checks `manifest.yaml` against the
  schema and verifies `ownership.team` / `ownership.poc` are
  not placeholders.
- `validate structure` — checks folder name, size, required
  files, and layout.
- `validate readme` — checks README.md for a setup section,
  run section, code block, and minimum word count.

### Format and lint (Python only)

- [ ] Format and lint pass:
      ```
      uv run ruff format $RECIPE_PATH
      uv run ruff check $RECIPE_PATH
      ```

### Tests (Python only)

- [ ] Tests pass (integration excluded, same as CI):
      ```
      cd $RECIPE_PATH
      uv run pytest --ignore=tests/integration --ignore-glob="**/test_integration.py"
      ```

### About integration tests

Integration tests exercise real external resources — Gemini,
BigQuery, third-party APIs, and similar. They're the most
valuable tests you can write before shipping a recipe: they
prove the whole thing actually works end-to-end. But they need
credentials, network access, and can be slow or flaky — so
GitHub CI skips them on every PR to keep the required check
fast and credential-free.

For Python recipes, CI treats these two patterns as integration
tests and excludes them:

- Anything under `tests/integration/`
- Any file named `test_integration.py`, at any depth

Run them locally before opening a PR if your recipe hits real
services:

    cd $RECIPE_PATH
    uv run pytest                      # full suite including integration

### Full manual command sequence

One block, one paste. Requires `$RECIPE_PATH` from the "Set
your recipe path" step above.

```bash
# Set your recipe path
export RECIPE_PATH=contrib/python/my-recipe # Replace with your recipe's folder name, not the recipe name

# From the repo root
uv run validate $RECIPE_PATH                # Validates the manifest and structure of the recipe
uv run ruff format $RECIPE_PATH             # Formats the recipe code
uv run ruff check --fix $RECIPE_PATH        # Fixes lint errors in the recipe code

# From the recipe root
cd $RECIPE_PATH
uv lock                                    # Updates the lock file 
uv run pytest                              # Runs the tests
```

## 4. When something fails

- CI red on your PR? →
  [troubleshooting](./recipe-handbook/troubleshooting.md)
- Not sure what a check is doing? →
  [ci-checks](./recipe-handbook/ci-checks.md)
- Want the full "how does it all fit together" story? →
  [handbook overview](./recipe-handbook/README.md)
