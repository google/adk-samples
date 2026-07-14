---
name: prepare-python-recipe
description: >
  End-to-end orchestration to prepare or update a Python recipe under
  core/python/ or contrib/ so it passes every check in
  .github/workflows/validate-python-recipe.yml. Runs six phases in order
  on an already-in-place recipe: manifest.yaml generation, environment-
  variable extraction, ruff format+check, pyproject.toml alignment, per-
  recipe `uv sync`, and runnability-test generation. Assumes the user has
  already done the manual prep (deactivated any venv, `git pull` and
  `uv sync` from the repo root, placed the recipe at its target path,
  renamed if needed). Delegates to the existing sub-skills
  (generate-manifest, extract-python-environment-variables,
  align-recipe-pyproject, generate-python-runnability-test) so the master
  never duplicates their logic. Pauses at fixed decision points (manifest
  team/POC verification, description mismatch, existing test regeneration)
  AND is free to interrupt for clarification any time a phase's output
  looks ambiguous, unexpected, or would benefit from a human judgment
  call — this is an interactive skill by design. Use when the user wants
  to "prepare a recipe", "update a recipe end to end", "run all the
  checks and fixes", "make this recipe PR-ready", or invokes it by name.
metadata:
  author: Google
  license: Apache-2.0
  version: 1.0.0
---

# Prepare Python Recipe

Master orchestration skill. Runs the other Python-recipe skills in the right order, with the right inputs, in a single pipeline. Use when the user wants a recipe brought fully up to standard in one go.

**This is an interactive skill.** It's expected to pause and ask questions when doing so genuinely improves the outcome — not just at the four fixed checkpoints below, but any time a phase's output is ambiguous, surprising, or would benefit from a judgment call. See rule 5 (fixed checkpoints) and rule 6 (judgment-based interruptions) for the difference.

---

## Prerequisites (manual, done by the user BEFORE invoking this skill)

The skill assumes the user has already:

1. **Deactivated** any active Python virtual environment.
2. **Pulled latest** from `origin` (`git pull` at the repo root).
3. **Synced repo root deps** (`uv sync` at the repo root).
4. **Placed the recipe at its target path** — either freshly scaffolded, moved from another location, or renamed to its final basename under `core/python/<name>/` or `contrib/<name>/`.

If the user has NOT done these and asks you to run the skill anyway, tell them to complete the prerequisites first and stop. Do NOT run `git pull`, deactivate their venv, or move/rename directories on their behalf — those are deliberately out of scope.

---

## What This Skill Does

Runs six ordered phases against a target recipe. Each phase either invokes an existing sub-skill (or its underlying script) or runs a repo-standard command:

1. **Manifest** — generate `manifest.yaml` if missing; ask the user to verify `ownership.team` and `ownership.poc`.
2. **Environment variables** — extract env vars used by the recipe into `.env.example`; ensure `load_dotenv()` is bootstrapped and `python-dotenv` is a dep.
3. **Lint** — `ruff format` + `ruff check --fix` on the recipe (from the repo root, so the root ruff config wins).
4. **Align pyproject.toml** — remove `[tool.ruff*]`, raise `requires-python` floor, ensure `[project].name` matches folder, reconcile description with manifest, and ensure `[[tool.uv.index]]` declares public PyPI as default (needed to bypass corp Airlock).
5. **Recipe `uv sync`** — sync the recipe's own venv after pyproject.toml is finalized.
6. **Runnability test** — generate `tests/test_runnability.py` if missing (or ask before overwriting).

At the end, print a summary table and remind the user to `git diff` and commit — the skill never commits.

---

## Rules for the Agent

1. **Ask for `--recipe-dir` up front** if the user hasn't given one. All six phases operate on the same recipe.

2. **Confirm before starting**. The pipeline touches many files. Show the user the plan (the six phases + the target recipe path) and ask for a single "yes, go ahead" before Phase 1. Do NOT prompt again for each phase unless a decision is required (see rules 5 and 6).

3. **Invoke sub-skill SCRIPTS directly** (not the sub-skills' own agent-facing SKILL.md). Reason: sub-skills each have their own "want me to apply?" prompt. In master-orchestration mode the user has already opted into apply for the whole pipeline; individual prompts would be noise. Command lines for each sub-script are given in each phase below.

4. **Exception for pure-instructions skills**: `generate-manifest` has no script — it's a pure-instructions skill. For that one only, load its SKILL.md (via the `skill` tool) and follow it inline.

5. **Fixed checkpoints — always pause here**:
   - **Manifest team/POC verification** — after Phase 1, `manifest.yaml` will contain the placeholders `"YOUR TEAM NAME"` and `"your-github-id"`. Show them and ask for real values.
   - **Description mismatch** — if Phase 4 returns `needs_input` for `description-matches-manifest`, show both sides and ask the user to pick `pyproject`, `manifest`, or `delete`.
   - **Test file exists** — before Phase 6, if `tests/test_runnability.py` already exists, ask whether to regenerate (default: keep existing). Regeneration uses `--overwrite`.
   - **Anything a sub-script flags as `error`** — surface the message, stop the pipeline, do NOT retry.

6. **Judgment-based interruptions — pause when it genuinely helps.** This skill is interactive by design. Beyond the fixed checkpoints in rule 5, feel free to interrupt any time doing so meaningfully improves the outcome. Some situations where a pause is appropriate:
   - A sub-script returns unexpected detections (e.g. the runnability-test generator reports `has_root_agent: false` — a legit recipe should always have one; something may be wrong).
   - The manifest generator inferred an `architecture.agent = "multi"` where you counted only one agent, or vice versa.
   - The environment-variable extractor added an unusually large number of new vars (say, ≥ 10) — worth having the user glance at the list.
   - The align script's proposed rewrite of `requires-python` drops support for a version the recipe's README claims to support.
   - `uv sync` in Phase 5 pulls in a suspicious dependency (say, one that renames or replaces a well-known package).
   - The recipe has a non-standard layout the sub-skills don't recognise (multiple `agent.py` files, no `app/` package, etc.) and you're unsure which to use.
   - Any time the "right answer" for a step depends on knowledge outside the recipe itself (project conventions, team decisions, downstream consumers).

   Do NOT interrupt for:
   - Progress updates ("Phase 3 done, moving to Phase 4?") — just move on.
   - Cosmetic curiosity ("I noticed a TODO in agent.py, want to discuss?") — out of scope.
   - "Just to make sure" prompts where the answer wouldn't change what you do next.

   When you interrupt, present the specific concern, show the relevant data, and offer concrete options — don't just say "does this look OK?".

7. **Halt on hard error**. If any phase's script exits with a non-zero code that isn't `refused_overwrite` (handled by rule 5), stop. Print the phase name, the error, and what's already been done. Do NOT continue past a hard error.

8. **Report progress compactly**. After each phase, one line: `Phase N (<name>): <one-line outcome>`. Do NOT dump raw JSON. Do NOT re-render each sub-skill's own table — the summary at the end covers it. Judgment-based interruptions from rule 6 are separate from progress lines and should be their own turn (question, then wait for the answer).

9. **Never commit**. The skill is done when the summary is printed. Let the user `git diff` and commit.

---

## Input

| Field | Required | Description |
|---|---|---|
| Recipe directory | Yes | Path to the recipe root (e.g. `core/python/cross-session-memory`, `contrib/my-recipe`). Passed to every sub-script as `--recipe-dir`. |

If the user has not specified the recipe directory, ask for it before proceeding.

---

## Pipeline

### Phase 0 — plan + confirm (do this first, always)

Before running anything, quickly confirm the prerequisites and show the user the plan:

> Prerequisites (I'll assume these are done — tell me if not):
>   - You've deactivated any active venv.
>   - You've run `git pull` and `uv sync` at the repo root.
>   - `<RECIPE_DIR>` is already at its target path (and renamed to its final basename).
>
> I'll run the prepare-python-recipe pipeline on `<RECIPE_DIR>` — 6 phases:
> 1. Generate manifest.yaml (if missing; verify team + POC)
> 2. Extract env vars into .env.example
> 3. Ruff format + check --fix
> 4. Align pyproject.toml
> 5. uv sync inside the recipe
> 6. Generate tests/test_runnability.py (if missing)
>
> Nothing gets committed — you'll `git diff` at the end. Proceed?

Get a yes-or-no. If no, stop.

### Phase 1 — manifest.yaml

**1a. Check whether manifest exists.**

```bash
[ -f <RECIPE_DIR>/manifest.yaml ] && echo exists || echo missing
```

**1b. If missing** — load the `generate-manifest` skill (via the `skill` tool with `name="generate-manifest"`) and follow its instructions for this recipe. That skill writes `manifest.yaml` with placeholders for team/POC.

**1b. If exists** — skip generation. Note it in the summary.

**1c. Verify team/POC** — regardless of whether we generated fresh or the file already existed, read `manifest.yaml` and locate `ownership.team` and `ownership.poc`. If either equals a placeholder value (`"YOUR TEAM NAME"` or `"your-github-id"`), pause and ask the user for real values. When they answer, edit `manifest.yaml` in place (use the `edit` tool). If both are already filled in, do not ask.

Progress line: `Phase 1 (manifest): generated | pre-existing; team=<x>, poc=<y>.`

### Phase 2 — extract env vars

Invoke the script directly:

```bash
python3 .agents/skills/extract-python-environment-variables/scripts/extract_env_vars.py \
  --recipe-dir <RECIPE_DIR>
```

(No `--dry-run` — master runs it in apply mode.)

The script:
- Appends any newly-discovered env vars to `.env.example`
- Injects `load_dotenv()` into the package `__init__.py` if missing
- Adds `python-dotenv>=1.0.0` to `[project].dependencies` if missing

Read the script's stdout (human-readable table). Extract the counts: how many vars added, whether `load_dotenv` was injected, whether `python-dotenv` was added.

Progress line: `Phase 2 (env vars): <N> vars added to .env.example, load_dotenv <injected|already present>, python-dotenv <added|already present>.`

### Phase 3 — ruff

Run both from the repo root (not the recipe dir) so the root `pyproject.toml`'s ruff config wins:

```bash
uv run ruff format <RECIPE_DIR>
uv run ruff check --fix <RECIPE_DIR>
```

If `ruff check` reports remaining un-fixable errors (exit non-zero), stop and surface them. If it fixes everything, continue.

Progress line: `Phase 3 (lint): <N> file(s) formatted, <M> issue(s) auto-fixed.`

### Phase 4 — align pyproject.toml

Invoke the align script directly. Two-pass logic:

**4a. Dry-run first** to detect whether description mismatch will need user input:

```bash
uv run --no-project --with tomlkit --with 'ruamel.yaml' --with packaging \
  python .agents/skills/align-recipe-pyproject/scripts/align_pyproject.py \
  --recipe-dir <RECIPE_DIR> --dry-run
```

Parse the JSON. If any check has `status == "needs_input"` for `description-matches-manifest`, pause: show both `pyproject_description` and `manifest_description` from `details`, ask the user to pick `pyproject`, `manifest`, or `delete`.

**4b. Apply** — with the description-source flag if you got one:

```bash
uv run --no-project --with tomlkit --with 'ruamel.yaml' --with packaging \
  python .agents/skills/align-recipe-pyproject/scripts/align_pyproject.py \
  --recipe-dir <RECIPE_DIR> \
  [--description-source=<CHOICE>]
```

If the apply run has any `report_only` results, note them in the summary — the master does NOT auto-fix these. Do not stop the pipeline; other phases continue. Two rules can produce `report_only`:
  - `build-system-present` (missing `[build-system]` — backend choice is editorial)
  - `default-pypi-index` (a default index is declared but points somewhere other than public PyPI — divergence may be intentional)

Progress line: `Phase 4 (align): <N> fix(es) applied; <M> report-only issue(s) left.`

### Phase 5 — recipe `uv sync`

Now that `pyproject.toml` is stable, sync the recipe's own venv:

```bash
uv sync
```

Run this WITH `workdir = <RECIPE_DIR>` (do not `cd` — pass the working directory via the tool call).

If `uv sync` fails (dependency conflict, missing package on PyPI, etc.), stop and surface the error.

Progress line: `Phase 5 (recipe sync): uv sync completed.`

### Phase 6 — runnability test

**6a. Check whether the test exists.**

```bash
[ -f <RECIPE_DIR>/tests/test_runnability.py ] && echo exists || echo missing
```

**6b. If missing** — generate:

```bash
python3 .agents/skills/generate-python-runnability-test/scripts/generate_runnability_test.py \
  --recipe-dir <RECIPE_DIR>
```

**6b. If exists** — pause and ask the user: keep existing (default) or regenerate. If they choose regenerate, run with `--overwrite`.

If the script errors (no `agent.py` found), surface the message and offer to re-run with `--agent-file <path>` when the user tells you where the entry point is.

Progress line: `Phase 6 (runnability test): generated | kept existing | regenerated.`

---

## Respond

While the pipeline runs, print a short progress line per phase (see above). Do NOT dump raw JSON. Do NOT re-render sub-skill tables.

At the end, print a summary table:

| Phase | Outcome | Notes |
|---|---|---|
| 1. Manifest | ok | generated; team=<x>, poc=<y> |
| 2. Env vars | ok | 3 added, load_dotenv injected, python-dotenv added |
| 3. Lint | ok | 12 files formatted, 4 issues auto-fixed |
| 4. Align | ok | 2 fixes applied |
| 5. Recipe sync | ok | done |
| 6. Runnability test | ok | generated |

Do NOT use emoji unless the user has asked for them. Use plain words in the Outcome column (`ok` / `skipped` / `failed`).

If any phase produced `report_only` items (typically `build-system-present`), list them as **Manual TODOs** below the table:

> Manual TODOs:
> - `[build-system]` is missing from `pyproject.toml`. Add one of the templates from the align-recipe-pyproject skill's SKILL.md (hatchling or uv_build).

Close with:

```
Next steps:
  cd <RECIPE_DIR>
  uv run pytest tests/test_runnability.py -v     # confirm the runnability test passes
  git diff                                        # review every change
  # commit when you're happy
```

Then stop. Do NOT commit. End your turn.
