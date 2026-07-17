---
name: extract-python-environment-variables
description: >
  Scans a Python recipe to find every place an environment variable is read,
  then ensures all variables are declared in .env.example, that load_dotenv()
  is bootstrapped in the package __init__.py, and that python-dotenv>=1.0.0
  is listed in pyproject.toml.   Also detects hardcoded model-name string literals in the recipe's source
  (e.g. "gemini-3.5-flash" in agent.py) and rewrites them to bare
  os.getenv("MODEL_NAME") calls (single model) or
  os.getenv("MODEL_NAME_GENERATED_1") / os.getenv("MODEL_NAME_GENERATED_2")
  etc. (multiple models) — NO fallback default in source. The actual model
  string is written as the value in .env.example (e.g.
  MODEL_NAME_GENERATED_1=gemini-3.5-flash) with a comment reminding the
  maintainer to rename the variable — this modifies source files, not just
  configuration. IMPORTANT — two hard rules the skill NEVER breaks:
  (1) Regular env vars in `.env.example` get ONLY TODO placeholders, never
  inferred defaults. Hardcoded model strings are the exception: their known
  value IS written to .env.example because it is not inferred — it was
  literally in the source.
  (2) Python source files get NO default values written by the skill —
  the model-replacement path emits `os.getenv("VAR")` with no second
  argument, and the skill never emits `os.environ.setdefault(...)`
  bootstrap lines. Pre-existing `os.environ.setdefault(...)` or
  `os.getenv("VAR", "default")` calls that the recipe author wrote by
  hand are LEFT UNTOUCHED — the skill is additive-only for Python files
  (adds `load_dotenv()` bootstrap; replaces hardcoded model literals;
  appends `# noqa: E402` to trailing relative imports in `__init__.py`
  when they'd otherwise trip Ruff after an env-bootstrap block).
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

   If `load_dotenv` is already present the injection is skipped.

   **Additionally — always, regardless of whether we injected** — appends
   `# noqa: E402 -- must come after load_dotenv()` to any top-level relative
   import (`from .x import y`) that sits AFTER a non-import module-level
   statement. Two cases this covers:

   - **Fresh injection.** The injected `load_dotenv()` call pushes
     pre-existing trailing relative imports below a non-import statement, so
     they'd trigger Ruff `E402` ("module-level import not at top of file")
     when Phase 4 (ruff) of `prepare-python-recipe` runs.

   - **Author-written bootstrap.** The recipe author already wrote
     `load_dotenv()` + `os.environ.setdefault(...)` calls followed by a
     trailing `from . import agent`, but never marked the trailing import.
     The skill did NOT inject anything (load_dotenv was already present) but
     still adds the noqa suffix so the file is lint-clean on the pipeline's
     next ruff pass.

   The suppression pass is precise — a relative import at the very TOP of
   the file (before any non-import statement) is fine and left untouched.
   Idempotent: a line that already carries `# noqa: E402` is skipped.

4. **Replaces hardcoded model names** in source (e.g. `model="gemini-3.5-flash"`
   in `agent.py`) with **bare `os.getenv(...)`** — no default argument:
   - Single model → `os.getenv("MODEL_NAME")`
   - Multiple models → `os.getenv("MODEL_NAME_GENERATED_1")`, `os.getenv("MODEL_NAME_GENERATED_2")`, … (sorted alphabetically for determinism)

   The actual model string is written as the value in `.env.example` (e.g.
   `MODEL_NAME_GENERATED_1=gemini-3.5-flash`) with a comment prompting the
   maintainer to rename the variable to something meaningful before shipping.

5. **Updates `pyproject.toml`** — adds `python-dotenv>=1.0.0` to `[project]`
   dependencies if it is not already there.

---

## Hard rules the skill NEVER breaks

**Rule 1 — No inferred defaults for regular env vars.** For variables read
via `os.getenv`/`os.environ`, `.env.example` always gets `<TODO: update-this-value>`,
even when the source has an `os.getenv("VAR", "some-fallback")`. Exception:
hardcoded model strings replaced in source ARE written with their actual value
in `.env.example` (e.g. `MODEL_NAME_GENERATED_1=gemini-3.5-flash`) — this is
not an inferred default, it is the known value that was literally in the source.
The source replacement always emits bare `os.getenv("VAR")` with **no second
argument** in both cases.

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
