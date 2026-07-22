<!-- word count: 885 (target 500+, no cap) -->

# Troubleshooting

Errors and warnings you'll see on a recipe PR, mapped to the
fix.

## `manifest.yaml missing` / `invalid per schema`

**Workflow:** `validate-recipe-structure.yml`.

- Missing entirely: run `generate-manifest`.
- Present but invalid: check against the
  [schema](../../.github/schemas/manifest-schema.json). Required
  fields: `type`, `status`, `language`, `description`,
  `ownership.team`, `ownership.poc`.

## `ownership.team is a placeholder`

**Workflow:** `validate-recipe-structure.yml`.

Values like `TODO`, `your-team`, `changeme`, or the literal
`<TODO>` fail. Set to a real team name.

## `Directory name exceeds 30 characters` / `does not match [a-z-]`

**Workflow:** `validate-recipe-structure.yml`.

Rename the folder. Lowercase letters and hyphens only, ≤ 30
chars, starts with a letter.

## `README.md is missing` / `is empty`

**Workflow:** `validate-recipe-structure.yml`.

Every recipe needs a `README.md` explaining what the recipe
does, how to set it up, and how to run it. Create the file and
cover at minimum: description, setup, run.

## `README.md contains TODO: placeholders`

**Workflow:** `validate-recipe-structure.yml`.

Replace every `TODO:` line with real content. Scaffold text and
draft notes must be resolved before submitting.

## `README.md is too short` (fewer than 100 words)

**Workflow:** `validate-recipe-structure.yml`.

The README must be at least 100 words. Add a real description,
setup instructions, and run steps. Aim for 200-300 words.

## `README.md is missing a setup section`

**Workflow:** `validate-recipe-structure.yml`.

Add a heading whose text contains one of: `Setup`,
`Prerequisites`, `Installation`, `Requirements`, `Configuration`,
`Getting Started`, `Before You Begin`, `Environment`.

## `README.md is missing a run section` / `has no fenced code block`

**Workflow:** `validate-recipe-structure.yml`.

Add a heading whose text contains one of: `Run`, `Running`,
`Usage`, `Quickstart`, `Start`, `Deploy`, `How to Run`. Under
that heading, include at least one fenced code block (```` ``` ````)
showing the exact command to start the agent.

Run `uv run validate readme <recipe-path>` locally to check
all six README rules before opening a PR.

## `Recipe exceeds size / file limit`

**Workflow:** `validate-recipe-structure.yml`.

- `contrib/` default: 70 files / 2 MB.

Fixes:

- Move data files (> 1 MB) to a linked storage bucket.
  Reference them in `README.md`.
- Delete generated files that shouldn't be committed (`.venv/`,
  IDE configs, build output — most of these are already
  excluded; check the workflow output for the actual counted
  paths).

## `pyproject.toml has [tool.ruff*] block`

**Workflow:** `python-validate-recipe.yml`.

Delete every `[tool.ruff]` / `[tool.ruff.*]` table from the
recipe's `pyproject.toml`. Ruff config is centralised in the
root `pyproject.toml`. Run `align-recipe-pyproject` to clean up.

## `Standalone ruff.toml / .ruff.toml found`

**Workflow:** `python-validate-recipe.yml`.

Same as above — delete the file. Ruff config is centralised.

## `env var used in source but missing from .env.example`

**Workflow:** `python-validate-recipe.yml`.

Run `extract-python-environment-variables`. It AST-parses your
source and adds every referenced env var to `.env.example`.

If the "env var" is a false positive (e.g. `os.getenv("HOME")`),
it should already be suppressed by the checker's ignore list.
If it isn't, file an issue against `check_env_vars.py`.

## `[project].name doesn't match folder name`

**Workflow:** `python-validate-recipe.yml`.

Set `[project].name` in `pyproject.toml` to the recipe folder
basename. A recipe at `contrib/python/my-recipe` needs
`name = "my-recipe"`.

## `[project].description doesn't match manifest.description`

**Workflow:** `python-validate-recipe.yml`.

Either:

- Delete `[project].description` from `pyproject.toml` (it's
  optional), or
- Copy `manifest.description` verbatim into
  `[project].description`.

## `requires-python permits versions below 3.11`

**Workflow:** `python-validate-recipe.yml`.

Set `requires-python = ">=3.11"` in `pyproject.toml`.

## `tests/test_runnability.py missing`

**Workflow:** `python-tests.yml`.

Run `generate-python-runnability-test`.

## `uv.lock out of sync`

**Workflow:** `python-dependency-policy.yml` and `python-tests.yml`.

`uv lock --check` must pass — every `uv.lock` must be in sync
with its sibling `pyproject.toml`. Run `uv lock` from the
recipe root:

    cd contrib/python/my-recipe
    uv lock

## `pyproject.toml has no sibling uv.lock`

**Workflow:** `python-dependency-policy.yml`.

Every `pyproject.toml` with a `[project]` table or `[tool.uv]`
section needs a `uv.lock` next to it. Run `uv lock` in that
directory.

## `Lockfile references a non-PyPI URL`

**Workflow:** `python-dependency-policy.yml`.

Every entry in `uv.lock` must resolve to `pypi.org` or
`files.pythonhosted.org`. Internal registries, GitHub Packages,
Artifactory, and similar sources are rejected. Replace the
dependency with a PyPI-published alternative.

## `VCS (git) dependency in uv.lock`

**Workflow:** `python-dependency-policy.yml`.

`source = { git = "..." }` entries are not allowed. Git
dependencies are non-reproducible and bypass registry trust.
Publish the package to PyPI, or use a PyPI equivalent.

## `Local path dependency in uv.lock`

**Workflow:** `python-dependency-policy.yml`.

`path`, `editable`, or `directory` sources only exist on the
committer's machine and cannot be resolved elsewhere. Exception:
uv's self-referential `editable = "."` entry for the workspace
root package is allowed. Remove all other local path sources.

## `Missing package hash in uv.lock`

**Workflow:** `python-dependency-policy.yml`.

Every distribution must include a `sha256` hash. Missing hashes
usually mean the lockfile was hand-edited or generated with a
different tool. Regenerate cleanly:

    cd contrib/python/my-recipe
    rm uv.lock
    uv lock

## `Ruff format` or `Ruff check` failed

**Workflow:** `python-format.yml`.

Auto-fix:

    uv run ruff format <recipe-path>
    uv run ruff check --fix <recipe-path>

Some issues need manual fixing — Ruff reports which and cites
the rule ID.

## Advisory notices (never fail CI)

- **`Hardcoded model name`** — replace with
  `os.getenv("MODEL_NAME")`. Run
  `extract-python-environment-variables`.
- **`GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION / MODEL_NAME
  missing from .env.example`** — add them if your recipe uses
  them.

## Something else

If your error isn't listed here, check the workflow's log for
the exact error message and the file the error references.
Then either:

- Use your browser or editor's search to find keywords from the
  error message in this page.
- Open a GitHub issue at
  [github.com/google/adk-samples/issues](https://github.com/google/adk-samples/issues)
  with the workflow name, error message, and link to the failed
  run.

---

← [Checklist](../recipe-checklist.md) · [Handbook](./README.md)
