---
name: extract-python-environment-variables
description: >
  Scans a Python recipe to find every place an environment variable is read,
  then ensures all variables are declared in .env.example, that load_dotenv()
  is bootstrapped in the package __init__.py, and that python-dotenv>=1.0.0
  is listed in pyproject.toml. Also detects hardcoded model-name string
  literals in the recipe's source (e.g. "gemini-3-flash-preview" in
  agent.py) and rewrites them to bare os.getenv("MODEL_NAME") calls (NO
  fallback default), adding MODEL_NAME to .env.example with a TODO
  placeholder — this modifies source files, not just configuration.
  IMPORTANT — two hard rules the skill NEVER breaks:
  (1) `.env.example` gets ONLY TODO placeholders, never inferred defaults.
  (2) Python source files get NO default values written by the skill —
  the model-replacement path emits `os.getenv("VAR")` with no second
  argument, and the skill never emits `os.environ.setdefault(...)`
  bootstrap lines. Pre-existing `os.environ.setdefault(...)` or
  `os.getenv("VAR", "default")` calls that the recipe author wrote by
  hand are LEFT UNTOUCHED — the skill is additive-only for Python files
  (adds `load_dotenv()` bootstrap; replaces hardcoded model literals).
  Use when the user wants to "extract env vars", "update .env.example",
  "add load_dotenv", "replace hardcoded model names", or "fix environment
  variables" in a Python recipe.
metadata:
  author: Google
  license: Apache-2.0
  version: 1.0.0
---

# Extract Python Environment Variables

Use this skill to ensure a Python recipe properly declares and loads all
environment variables it uses.

---

## What This Skill Does

Runs `scripts/extract_env_vars.py` against a recipe directory. The script:

1. **Scans** all `.py` files (excluding `tests/`) for environment variable reads:
   - `os.environ["VAR"]`
   - `os.environ.get("VAR")` / `os.environ.get("VAR", "default")`
   - `os.getenv("VAR")` / `os.getenv("VAR", "default")`

   Only names matching `^[A-Z_][A-Z0-9_]*$` (UPPER_SNAKE_CASE) are captured;
   a lowercase name like `os.getenv("my_api_key")` is **skipped** and a
   `[WARN]` line lists any that were dropped. Rename such vars to uppercase
   in source, or add them to `.env.example` by hand.

2. **Updates `.env.example`** — appends any variables not already declared.
   - **Every value is the placeholder `<TODO: update-this-value>`.** The
     skill never writes inferred defaults into `.env.example`, even when
     the source code has an `os.getenv("VAR", "some_default")` fallback.
     `.env.example` is a template the human fills in; the source-code
     fallback is a runtime default, not a value the maintainer has
     committed to as canonical.
   - Creates `.env.example` from scratch if it does not exist.

3. **Injects `load_dotenv()`** into the package `__init__.py` (the first
   subdirectory inside the recipe that contains an `__init__.py`, skipping
   `tests/` and hidden directories). The snippet injected is:

   ```python
   from dotenv import load_dotenv

   # Load variables from .env if present. In production the environment is
   # already populated by the platform (Cloud Run, GKE, etc.), so a missing
   # .env is expected and not an error.
   load_dotenv()
   ```

   If `load_dotenv` is already present the file is left unchanged.

4. **Replaces hardcoded model names** in source (e.g. `model="gemini-3.5-flash"`
   in `agent.py`) with **bare `os.getenv("MODEL_NAME")`** — no default argument.

5. **Updates `pyproject.toml`** — adds `python-dotenv>=1.0.0` to `[project]`
   dependencies if it is not already there.

---

## Hard rules the skill NEVER breaks

**Rule 1 — No inferred defaults anywhere.** The skill never persists a
concrete default value it inferred from source code. Applies to:
- `.env.example` — every new entry gets `<TODO: update-this-value>`,
  even when the source has `os.getenv("VAR", "some-fallback")`.
- Python files — the model-replacement path emits
  `os.getenv("MODEL_NAME")` with **no second argument**. It does NOT emit
  `os.getenv("MODEL_NAME", "the-original-hardcoded-value")` even though
  that would be trivially "safer."

**Rule 2 — Additive-only for Python files.** The skill never writes new
`os.environ.setdefault(...)` bootstrap lines into any Python file.
Pre-existing `os.environ.setdefault(...)` calls or
`os.getenv("VAR", "default")` calls that a recipe author wrote by hand
are LEFT UNTOUCHED. The skill's only writes to Python files are:
- Adding the `from dotenv import load_dotenv` + `load_dotenv()` snippet
  (once, only if not already present).
- Replacing hardcoded model literals with bare `os.getenv(...)` calls.

Rationale: default values are the maintainer's decision, not the
skill's. Persisting an inferred default (in `.env.example` or in a new
Python bootstrap line) is presumptuous — it makes the recipe look like
it works when the value is really the maintainer's job to fill in.
Pre-existing lines are the maintainer's own committed choice; the
skill respects that and doesn't rewrite them.

---

## Rules for the Agent

1. **Always use the script** — never manually edit `.env.example`, `__init__.py`,
   or `pyproject.toml` to perform these changes.
2. **Ask for the recipe directory** if the user has not provided one. Do not
   assume a path.
3. **After the script succeeds**, remind the user to:
   - Copy `.env.example` → `.env` and fill in real values before running locally.
   - Run `uv sync` to pick up the `python-dotenv` dependency if it was newly added.

---

## Input

| Field | Required | Description |
|-------|----------|-------------|
| Recipe directory | Yes | Path to the recipe root (e.g. `contrib/my-recipe` or `core/python/my-recipe`) |

If the user has not specified the recipe directory, ask for it before proceeding.

---

## Run

Run it through `uv` so it always executes on a Python 3.11+ interpreter — the
script uses the stdlib `tomllib`, which only exists from 3.11 onward. A bare
`python3` that resolves to 3.9/3.10 fails with `ModuleNotFoundError: tomllib`.
No `--with` packages are needed (the script is stdlib-only).

```bash
uv run --no-project python3 \
  .agents/skills/extract-python-environment-variables/scripts/extract_env_vars.py \
  --recipe-dir <RECIPE_DIR>
```

### Preview first (optional)

Add `--dry-run` to report exactly what *would* change without modifying any
files. Nothing is written to `.env.example`, `__init__.py`, `pyproject.toml`, or
any source file. Useful for inspecting a recipe before committing to the edits:

```bash
uv run --no-project python3 \
  .agents/skills/extract-python-environment-variables/scripts/extract_env_vars.py \
  --recipe-dir <RECIPE_DIR> --dry-run
```

In dry-run output, actions are prefixed with `[DRY-RUN]` and phrased as
"Would add" / "Would inject" / "Would replace".

---

## Respond

Do not show the script's raw stdout. Reformat its results into clear Markdown
tables (variables to add, files to update, model replacements) so they are easy
to read — this is especially important for `--dry-run` output. For each variable
or model string, include a column with the source file where it was found
(locate it in the recipe's Python source; ignore `.env` and `.env.example`).

Once the script finishes successfully, summarise what changed:

- Which variables were added to `.env.example` (or confirm it was already up to date).
- Whether `load_dotenv()` was injected or was already present.
- Whether `python-dotenv` was added to `pyproject.toml` or was already there.

Then remind the user of the next steps:

```
Next steps:
  cp <RECIPE_DIR>/.env.example <RECIPE_DIR>/.env   # then fill in real values
  cd <RECIPE_DIR> && uv sync                        # install python-dotenv if newly added
```

Do not make any further changes. End your turn.
