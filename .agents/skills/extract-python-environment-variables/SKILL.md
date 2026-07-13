---
name: extract-python-environment-variables
description: >
  Scans a Python recipe to find every place an environment variable is read,
  then ensures all variables are declared in .env.example, that load_dotenv()
  is bootstrapped in the package __init__.py, and that python-dotenv>=1.0.0
  is listed in pyproject.toml. Use when the user wants to "extract env vars",
  "update .env.example", "add load_dotenv", or "fix environment variables" in
  a Python recipe.
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

2. **Updates `.env.example`** — appends any variables not already declared.
   - If a default value is present in the code (e.g. `os.getenv("FOO", "bar")`),
     that value is used in `.env.example`.
   - Otherwise the placeholder `<UPDATE_THIS_VALUE>` is used.
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

4. **Updates `pyproject.toml`** — adds `python-dotenv>=1.0.0` to `[project]`
   dependencies if it is not already there.

---

## Rules

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

```bash
python3 .agents/skills/extract-python-environment-variables/scripts/extract_env_vars.py \
  --recipe-dir <RECIPE_DIR>
```

### Preview first (optional)

Add `--dry-run` to report exactly what *would* change without modifying any
files. Nothing is written to `.env.example`, `__init__.py`, `pyproject.toml`, or
any source file. Useful for inspecting a recipe before committing to the edits:

```bash
python3 .agents/skills/extract-python-environment-variables/scripts/extract_env_vars.py \
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
