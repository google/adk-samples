<!-- word count: 538 (target 800, cap 1200) -->

# Workflows

> **Before you start.** Every recipe must have a clear intent
> and deliver real value to the ADK community. State in one
> sentence what problem your recipe solves for developers using
> ADK — if you cannot, revisit the idea before writing code.
> Recipes that duplicate existing examples without new insight
> may not be accepted.

End-to-end scenarios. Each ties multiple checklist items and AI
skills together.

**Two paths through every scenario:**

- **With an AI assistant** (e.g.
  [Antigravity CLI](https://antigravity.google/product/antigravity-cli)):
  invoke skills by name — `prepare-python-recipe`,
  `align-recipe-pyproject`, etc.
- **Without an AI assistant:** run the equivalent commands
  manually. Every skill is a wrapper over commands you can run
  yourself; see
  [languages/python.md#local-commands](./languages/python.md#local-commands)
  for the manual sequence.

## New Python recipe from scratch

Steps from an empty folder to a passing CI check.

1. **Scaffold.** Ask your assistant:

       "scaffold a new Python recipe at contrib/python/my-recipe"

   Runs `scaffold-python-recipe`. Creates the folder with
   `pyproject.toml`, `app/`, `README.md`,
   `tests/test_runnability.py`, `.env.example`, and
   `manifest.yaml`.

2. **Write the agent.** Edit `app/agent.py`. Add tools,
   prompts, sub-agents. Reference:
   - [ADK developer docs](https://github.com/google/adk-python)
     for the agent API.
   - [languages/python.md](./languages/python.md) for the
     package layout this repo expects.
   - Existing recipes under `contrib/python/` for realistic
     examples.

3. **Prepare.** When the agent works locally:

       "prepare the python recipe contrib/python/my-recipe"

   Runs `prepare-python-recipe`: generates the manifest,
   extracts env vars, aligns `pyproject.toml`, runs Ruff, locks
   deps, generates the runnability test.

4. **Verify.** From the repo root:

       uv run validate contrib/python/my-recipe
       cd contrib/python/my-recipe && uv run pytest \
         --ignore=tests/integration --ignore-glob="**/test_integration.py"

   The runnability test (`tests/test_runnability.py`) loads
   `app/agent.py` and asserts `root_agent is not None`. It
   confirms your agent code does not crash on import — nothing
   more. Real API calls are not made; import-time side effects
   like `vertexai.init` are mocked.

5. **Open the PR.** CI runs the same checks and merges when
   all checks pass.

## Updating an existing recipe

**Small change** (bug fix, new tool, minor prompt adjustment):

1. Make the change.
2. Re-run the affected step:

   | If you changed... | Run this AI skill | Or run manually |
   |---|---|---|
   | An env var reference | `extract-python-environment-variables` | Update `.env.example` by hand |
   | `pyproject.toml` dependencies | (handled by `prepare-python-recipe`) | `uv lock` from the recipe root |
   | The agent's imports or top-level structure | `generate-python-runnability-test` | Regenerate `tests/test_runnability.py` by hand |
   | Any Python source | (handled by `prepare-python-recipe`) | `uv run ruff format <path>` and `uv run ruff check <path>` |

3. Re-run local verify commands (step 4 above).
4. PR.

All steps are covered when you run `prepare-python-recipe`
end-to-end.

**Large change** (rewrite, dependency bump): run
`prepare-python-recipe` end-to-end. It is safe to re-run — it
does not overwrite user-authored `.env.example` entries or
hand-written Python code.

## Adding an integration test that hits real GCP

Integration tests are excluded from CI (see
[languages/python.md#integration-tests](./languages/python.md#integration-tests)).
Add them for local and manual runs.

1. Create `tests/integration/test_<feature>.py` (or any file
   named `test_integration.py` at any depth).
2. Write the test — real API calls, real credentials.
3. Document credential setup in the recipe's `README.md`.
4. Run locally:

       cd contrib/python/my-recipe
       uv run pytest tests/integration

CI won't run them; that's intentional. Include a note in the PR
description if they passed locally.

---

← [Checklist](../recipe-checklist.md) · [Handbook](./README.md)
