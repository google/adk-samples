---
name: align-recipe-pyproject
description: >
  Aligns a Python recipe's pyproject.toml with the repo's standards enforced
  by .github/workflows/python-validate-recipe.yml, plus one critical
  [build-system] presence check. Scope is pyproject.toml only — standalone
  ruff.toml / .ruff.toml files (also forbidden in recipes) are caught by the
  CI workflow instead, not by this skill. Runs in two modes: a read-only
  --dry-run that reports what needs alignment, and an apply mode that
  rewrites pyproject.toml (and optionally manifest.yaml) using
  comment-preserving TOML/YAML editors. Use when the user wants to "align
  the recipe's pyproject.toml", "fix pyproject to match the repo standard",
  "check what needs changing in a recipe's pyproject", or clean up a recipe
  before submitting a PR.
metadata:
  author: Google
  license: Apache-2.0
  version: 1.0.0
---

# Align Recipe pyproject.toml

Use this skill to bring a Python recipe's `pyproject.toml` (and, when it disagrees, its `manifest.yaml` description) into conformance with the repo standard enforced by `.github/workflows/python-validate-recipe.yml`.

Scope: **`pyproject.toml` only**. Standalone `ruff.toml` / `.ruff.toml` files are also forbidden in recipes but are enforced by the CI workflow (Check 7 in `python-validate-recipe.yml`) — outside this skill's concern.

---

## What This Skill Checks

Runs `scripts/align_pyproject.py` against a recipe directory. Six rules:

| Rule ID | What it checks | Auto-fix |
|---|---|---|
| `no-local-ruff-config` | Recipe `pyproject.toml` must not declare any `[tool.ruff*]` table. Ruff config is centralized in the root `pyproject.toml`. | Yes — removes the tables. |
| `python-version-floor` | `[project].requires-python` must not permit any Python version below 3.11 (per `AGENTS.md`). A recipe that requires Python 3.12+ is the author's choice and is left alone. | Yes — raises the lower bound to `>=3.11` while preserving every upper bound, exclusion, compatible-release (`~=`) ceiling, and pin (only pure `>=`/`>` are dropped). If the result would be self-contradictory because the recipe's own ceiling/pin/exclusion excludes 3.11 (e.g. `>=3.10,!=3.11` or `==3.10.*`), refuses to apply and returns `needs_input`. |
| `project-name-matches-folder` | `[project].name` must equal the recipe folder basename. | Yes — sets it. |
| `description-matches-manifest` | If `[project].description` is set, it must equal `manifest.description`. Field is optional; skipped when absent. | Only with `--description-source={pyproject,manifest,delete}`. Refuses to touch description otherwise. |
| `build-system-present` | `[build-system]` must have both `requires` and `build-backend`. Without it, `uv build` and `pip install .` fail. | **No** — backend choice is editorial. Reported for the human to fix. |
| `default-pypi-index` | `[[tool.uv.index]]` must have an entry with `default = true` pointing at public PyPI (`https://pypi.org/simple[/]`). Required so `uv sync` works on Google corp workstations without corp Airlock auth — see the block comment in the root `pyproject.toml` for the full rationale. | Yes when the block is entirely missing — appends it. **No** when a default entry exists but points elsewhere (custom private index, TestPyPI, mirror) — reported for the human to reconcile, since the divergence may be intentional. |

### Edit safety

- TOML edits go through **`tomlkit`** — comments, blank lines, and unrelated tables in `pyproject.toml` are preserved.
- YAML edits (only when `--description-source=pyproject` overwrites `manifest.description`) go through **`ruamel.yaml`** — comments in `manifest.yaml` are preserved.
- Files are only rewritten if at least one auto-fix actually ran. Otherwise on-disk bytes are untouched.
- No files outside the recipe directory are ever modified. The workflow file, root `pyproject.toml`, and other recipes are safe.

---

## Rules for the Agent

1. **Always use the script — never hand-edit `pyproject.toml` or `manifest.yaml` to perform these changes.** The script exists specifically so edits are style-preserving and reviewable via one report.

2. **Ask for the recipe directory** if the user has not provided one. Do not guess. Recipe roots live under `core/python/<name>/` or `contrib/<name>/`.

3. **Always start with `--dry-run`** unless the user has explicitly said "apply", "fix it", "just do it", or equivalent. Show them what would change before doing it.

4. **Only report checks that need attention.** The script always returns every check's status, but the user does not need to see the ones that passed. Filter to `status != "ok"`.
   - If **nothing** needs attention: reply with a single positive line (see the response format below). Don't list what's fine, don't build a table.
   - If **something** needs attention: render only the non-ok checks.

5. **If `description-matches-manifest` returns `needs_input`**, the descriptions in `pyproject.toml` and `manifest.yaml` disagree and the script cannot pick a winner. Present the two values side by side and ask the user to choose one of the three resolution options. Do not re-run the script until they choose.

6. **If `build-system-present` returns `report_only`**, `[build-system]` is missing or incomplete. Tell the user the recipe cannot be built as a package until this is fixed, and offer them a hatchling or `uv_build` template snippet. Do **not** silently pick one.

7. **If `default-pypi-index` returns `report_only`**, the recipe declares a default index that is NOT public PyPI (e.g. a private mirror, TestPyPI). The skill will not overwrite an intentional choice. Show the user the current `url` from `details.current_url` and ask whether it's deliberate. If yes, they can `# noqa`-comment it or update the repo standard; if no, they should change the URL to `https://pypi.org/simple/`. Do not auto-rewrite.

8. **After apply mode succeeds**, remind the user to run `uv sync` in the recipe directory if the pyproject changes touched dependencies (the script emits this in `notes` when relevant).

9. **Do not commit any changes.** Show the diff or file contents; let the user commit.

---

## Input

| Field | Required | Description |
|---|---|---|
| `--recipe-dir` | Yes | Path to the recipe root (e.g. `core/python/cross-session-memory`, `contrib/my-recipe`). |
| `--dry-run` | No | Report what would change without modifying any files. |
| `--description-source` | Only when resolving a `description-matches-manifest` mismatch. Values: `pyproject`, `manifest`, `delete`. See below. | Chooses how to reconcile a description mismatch. |

### `--description-source` semantics

| Value | Effect |
|---|---|
| `pyproject` | Overwrite `manifest.description` with the value from `[project].description`. Use when pyproject is authoritative (e.g. it was updated more recently). |
| `manifest` | Overwrite `[project].description` with the value from `manifest.description`. Use when the manifest is authoritative. |
| `delete` | Remove `[project].description` from `pyproject.toml` entirely, so `manifest.description` becomes the single source of truth. Use when the recipe doesn't need a wheel-metadata description. |

If the descriptions already match, this flag is ignored.

---

## Run

### Dry-run (read-only report — start here)

```bash
uv run --no-project --with tomlkit --with 'ruamel.yaml' --with packaging \
  python .agents/skills/align-recipe-pyproject/scripts/align_pyproject.py \
  --recipe-dir <RECIPE_DIR> --dry-run
```

Output: JSON on stdout. Every check produces one entry with `id`, `status` (`ok` / `would_fix` / `needs_input` / `report_only` / `error`), a human-readable `message`, and structured `details`. Exit code is always `0` in dry-run.

### Apply (rewrite files)

```bash
uv run --no-project --with tomlkit --with 'ruamel.yaml' --with packaging \
  python .agents/skills/align-recipe-pyproject/scripts/align_pyproject.py \
  --recipe-dir <RECIPE_DIR>
```

Fixes everything the script can safely fix. Exits `0` if nothing is left unresolved, `1` otherwise (typically because `description-matches-manifest` needs `--description-source` or `build-system-present` needs a human).

### Apply with description resolution

```bash
uv run --no-project --with tomlkit --with 'ruamel.yaml' --with packaging \
  python .agents/skills/align-recipe-pyproject/scripts/align_pyproject.py \
  --recipe-dir <RECIPE_DIR> --description-source={pyproject|manifest|delete}
```

### Preview a description resolution (dry-run + `--description-source`)

`--description-source` combines with `--dry-run`: the script reports the
`would_fix` outcome of the chosen resolution **without writing any files**.
Use this to show the user exactly what a given choice will do before you
apply it — the two flags are not mutually exclusive.

```bash
uv run --no-project --with tomlkit --with 'ruamel.yaml' --with packaging \
  python .agents/skills/align-recipe-pyproject/scripts/align_pyproject.py \
  --recipe-dir <RECIPE_DIR> --dry-run \
  --description-source={pyproject|manifest|delete}
```

---

## Respond

Filter the report to checks with `status != "ok"`. Never list what's fine — the user is here to see what needs doing.

### If everything passed

Reply with a single line, no table, no bullets. Something like:

> ✓ `<RECIPE_DIR>`'s pyproject.toml is fully aligned with the repo standard. No changes needed.

Then stop. Do not run any further tools.

### If there are things to report

Name the recipe directory once, then render **a single Markdown table** with these three columns:

| Rule | Status | Details |
|---|---|---|

- **Rule** — the check `id`, in backticks (e.g. `` `no-local-ruff-config` ``).
- **Status** — the raw status value (`would_fix`, `fixed`, `needs_input`, `report_only`, `error`). Do not add emoji unless the user has asked for them.
- **Details** — a compact summary of both what's wrong and what will happen (or what the user must do), merged into one cell so the table stays 3 columns wide. Quote `from` → `to` values from `details` when present.

Include exactly one row per non-ok check. Do not include `ok` rows.

Status-specific guidance for what to put in the **Details** cell:

- **`would_fix`** (dry-run) — describe the current-state problem, then say what apply would do. Include the `from` → `to` or the list of tables to be removed.
- **`fixed`** (apply) — one-liner confirming the change (new value or list of removed tables).
- **`needs_input`** (only `description-matches-manifest`) — the Details cell says something like `"descriptions differ — needs --description-source={pyproject,manifest,delete}"`. Do not put the two long descriptions inside the table. See "Follow-up content" below.
- **`report_only`** (two rules can hit this: `build-system-present` and `default-pypi-index` when a non-PyPI default is declared) — the Details cell names what's missing or non-conforming (e.g. `"[build-system] missing; recipe cannot be built as a package"` or `"default index is TestPyPI, not public PyPI"`). Follow-up content goes below the table (see next section).
- **`error`** — the Details cell shows the message verbatim; if it's very long, truncate with `…` and put the full text below.

### Follow-up content below the table

Only for statuses that need extra context. Order: table first, then this content, then closing action.

- **`needs_input` follow-up** — present `pyproject_description` and `manifest_description` from `details` side by side (a small nested table works well), then ask the user to pick one of `pyproject`, `manifest`, or `delete`. Do not re-run until they answer.
- **`report_only` follow-up** — explain the impact (recipe cannot be built as a package) and paste both templates so the user can choose:

  ```toml
  # Hatchling (traditional Python packaging)
  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"

  [tool.hatch.build.targets.wheel]
  packages = ["app"]

  # ---- OR ----

  # uv_build (Astral's uv-native backend)
  [build-system]
  requires = ["uv_build>=0.8.14,<0.9.0"]
  build-backend = "uv_build"

  [tool.uv.build-backend]
  module-root = ""
  module-name = "app"
  ```

- **`error` follow-up** — if the message was truncated in the table, print it verbatim in full below. Do not attempt to work around it.

### Closing action

- **Dry-run with at least one `would_fix` row** — **offer to apply the fixes yourself.** Do NOT paste the raw command as a copy-and-paste snippet for the user. Instead ask something like "Want me to apply these fixes?" (a yes/no question is fine; a question with clear options is nicer UX). If the user agrees, run apply mode yourself and render the result as another table (with `fixed` rows instead of `would_fix`). If the user declines, stop.

- **Dry-run with only `needs_input` for `description-matches-manifest`** — after the table, show the two descriptions side by side and ask the user to pick one of `pyproject`, `manifest`, or `delete`. When they choose, run apply yourself with `--description-source=<their choice>` and render the resulting table. Do not offer generic "apply" — the flag is required.

- **Dry-run with only `report_only`** (and no `would_fix` rows) — after the table, address the specific case:
  - `build-system-present`: show the two `[build-system]` template snippets and stop. This is a manual edit; the skill does not auto-fix it.
  - `default-pypi-index`: quote `details.current_url`, explain that this is not public PyPI, and ask whether it's intentional. If not, tell the user to change the URL to `https://pypi.org/simple/`. Do not auto-rewrite.

- **Dry-run with only `error` rows** — do not offer to apply. Errors mean the script bailed before it could compute a fix; the user has to resolve the underlying issue first.

- **Dry-run with a mix** (e.g. some `would_fix` + a `needs_input`) — offer to apply the fixable ones. The apply run will fix those and leave the others as-is; render the resulting table and then address the remaining statuses per the rules above.

- **Apply mode with any `fixed` row** — end with the "Next steps" reminder (see below).

Do NOT commit any changes yourself. Ever.

### After apply mode

If any check has `status: fixed`, remind the user:

```
Next steps:
  cd <RECIPE_DIR> && uv sync    # if dependencies changed
  git diff                       # review the edits before committing
```

Then stop. Do not commit. Do not run any further tools. End your turn.
