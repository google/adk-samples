<!-- word count: 1200 (target 500+, no cap) -->

# Troubleshooting

Errors and warnings on a recipe PR, mapped to the fix.

**Re-running CI:** CI triggers automatically on every push to your
PR branch. No manual action is needed. If a check appears stuck,
a reviewer can re-run it from the GitHub Actions UI.

**Fix structural errors first.** Failures in `validate-recipe-structure`
can mask downstream Python checks. A stale `uv.lock` causes both
`python-dependency-policy` and `python-tests` to fail — run `uv lock`
once to clear both.

**Can't find your error?** Search this page for keywords from the
CI log, or [jump to Something else](#something-else).

## Contents

**Structure**
- [manifest.yaml missing or invalid](#manifestyaml-missing-or-invalid)
- [ownership.team or poc is a placeholder](#ownershipteam-or-poc-is-a-placeholder)
- [Directory name too long or invalid](#directory-name-too-long-or-invalid)
- [Recipe exceeds size or file limit](#recipe-exceeds-size-or-file-limit)

**README.md**
- [README.md is missing or empty](#readmemd-is-missing-or-empty)
- [README.md contains TODO placeholders](#readmemd-contains-todo-placeholders)
- [README.md is too short](#readmemd-is-too-short)
- [README.md is missing a setup section](#readmemd-is-missing-a-setup-section)
- [README.md is missing a run section or code block](#readmemd-is-missing-a-run-section-or-code-block)

**pyproject.toml**
- [pyproject.toml has a local ruff configuration](#pyprojecttoml-has-a-local-ruff-configuration)
- [Standalone Ruff config file](#standalone-ruff-config-file)
- [Project name doesn't match folder name](#project-name-doesnt-match-folder-name)
- [Project description doesn't match manifest](#project-description-doesnt-match-manifest)
- [requires-python below 3.11](#requires-python-below-311)
- [pyproject.toml has no sibling uv.lock](#pyprojecttoml-has-no-sibling-uvlock)

**Environment variables**
- [Env var missing from .env.example](#env-var-missing-from-envexample)

**Dependencies (uv.lock)**
- [uv.lock out of sync](#uvlock-out-of-sync)
- [Lockfile references a non-PyPI URL](#lockfile-references-a-non-pypi-url)
- [VCS dependency in uv.lock](#vcs-dependency-in-uvlock)
- [Local path dependency in uv.lock](#local-path-dependency-in-uvlock)
- [Missing package hash in uv.lock](#missing-package-hash-in-uvlock)

**Tests and code style**
- [Runnability test missing](#runnability-test-missing)
- [Ruff format or check failed](#ruff-format-or-check-failed)

**Other**
- [Non-blocking notices](#non-blocking-notices)
- [Something else](#something-else)

---

## manifest.yaml missing or invalid

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `[manifest] manifest.yaml is missing.` or `[manifest] <schema error>`

If `manifest.yaml` is missing entirely, run `generate-manifest` (AI
skill). If you don't have an AI coding assistant, create the file by
hand — see the [minimum example in anatomy.md](./anatomy.md#manifestyaml).

If the file is present but failing schema validation, check it
against the [schema](../../.github/schemas/manifest-schema.json).
Required fields: `type`, `status`, `language`, `description`,
`ownership.team`, `ownership.poc`.

## ownership.team or poc is a placeholder

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `[ownership.team] is still set to the placeholder value` or `[ownership.poc] is still set to the placeholder value`

Replace the placeholder value in `manifest.yaml` with a real team
name (`ownership.team`) and a real GitHub user ID (`ownership.poc`).

## Directory name too long or invalid

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `[folder-name] Folder name`

Rename the recipe folder. Rules: lowercase letters and hyphens only,
starts with a letter, maximum 30 characters (`^[a-z][a-z-]*$`).

## README.md is missing or empty

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `README.md is missing` or `README.md is empty`

Every recipe needs a `README.md` that explains what the recipe does,
how to set it up, and how to run it. Create the file and cover at
minimum: description, setup, and run.

## README.md contains TODO placeholders

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `README.md contains TODO: placeholders`

Replace every `TODO:` line with real content. Scaffold text and draft
notes must be resolved before opening a PR.

## README.md is too short

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `README.md is too short`

The README must be at least 100 words. Add a real description, setup
instructions, and run steps. Aim for 200–300 words.

## README.md is missing a setup section

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `README.md is missing a setup section`

Add a heading whose text contains one of: `Setup`, `Prerequisites`,
`Installation`, `Requirements`, `Configuration`, `Getting Started`,
`Before You Begin`, `Environment`.

## README.md is missing a run section or code block

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `README.md is missing a run section` or `README.md has no fenced code block`

Add a heading whose text contains one of: `Run`, `Running`, `Usage`,
`Quickstart`, `Start`, `Deploy`, `How to Run`. Under that heading,
include at least one fenced code block (```` ``` ````) showing the
exact command to start the agent.

Run `uv run validate readme <recipe-path>` locally to check all
README rules before opening a PR.

## Recipe exceeds size or file limit

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `Recipe folder exceeds` or `exceeding the limit`

`contrib/` default limits: 70 files / 2 MB.

To fix: move data files larger than 1 MB to a linked storage bucket
and reference them in `README.md`. Delete generated files that
shouldn't be committed (`.venv/`, IDE configs, build output — check
the workflow output for the actual counted paths).

## pyproject.toml has a local ruff configuration

**Workflow:** [`python-validate-recipe.yml`](../../.github/workflows/python-validate-recipe.yml)

CI output contains: `pyproject.toml contains a [tool.ruff*] block`

Delete every `[tool.ruff]` and `[tool.ruff.*]` table from the
recipe's `pyproject.toml`. Ruff configuration is centralized in the
root `pyproject.toml`. Run `align-recipe-pyproject` (AI skill) to
clean it up automatically.

## Standalone Ruff config file

**Workflow:** [`python-validate-recipe.yml`](../../.github/workflows/python-validate-recipe.yml)

CI output contains: `Standalone Ruff config found`

Delete the `ruff.toml` or `.ruff.toml` file from the recipe
directory. Ruff configuration is centralized in the root
`pyproject.toml`.

## Env var missing from .env.example

**Workflow:** [`python-validate-recipe.yml`](../../.github/workflows/python-validate-recipe.yml)

CI output contains: `Environment variable '<VAR>' is read by Python source but not declared in .env.example`

Run `extract-python-environment-variables` (AI skill). It parses
your Python source and adds every referenced env var to `.env.example`.

If you don't have an AI coding assistant, add the missing variable
name to `.env.example` manually. The CI error message lists the exact
variable name(s).

If the reported variable is a false positive (for example,
`os.getenv("HOME")`), it should already be suppressed by the
checker's ignore list. If it isn't, file an issue against
[`.github/scripts/check_env_vars.py`](../../.github/scripts/check_env_vars.py).

## Project name doesn't match folder name

**Workflow:** [`python-validate-recipe.yml`](../../.github/workflows/python-validate-recipe.yml)

CI output contains: `[project].name` and `does not match the recipe folder name`

Set `[project].name` in `pyproject.toml` to the recipe folder
basename. A recipe at `contrib/python/my-recipe` needs:

```toml
name = "my-recipe"
```

## Project description doesn't match manifest

**Workflow:** [`python-validate-recipe.yml`](../../.github/workflows/python-validate-recipe.yml)

CI output contains: `[project].description does not match manifest.description`

Either delete `[project].description` from `pyproject.toml` (it is
optional), or copy `manifest.description` verbatim into
`[project].description`.

## requires-python below 3.11

**Workflow:** [`python-validate-recipe.yml`](../../.github/workflows/python-validate-recipe.yml)

CI output contains: `[project].requires-python` and `permits Python versions below 3.11`

Set the minimum Python version in `pyproject.toml`:

```toml
requires-python = ">=3.11"
```

## Runnability test missing

**Workflow:** [`python-tests.yml`](../../.github/workflows/python-tests.yml)

CI output contains: `No test_runnability.py found under`

Run `generate-python-runnability-test` (AI skill).

If you don't have an AI coding assistant, copy the minimal template
from [python.md — Copy-paste starters](./languages/python.md#copy-paste-starters)
and adjust it for your agent.

## uv.lock out of sync

**Workflow:** [`python-dependency-policy.yml`](../../.github/workflows/python-dependency-policy.yml) and [`python-tests.yml`](../../.github/workflows/python-tests.yml)

CI output contains: `is out of date — run: uv lock`

`uv lock --check` must pass — every `uv.lock` must be in sync with
its sibling `pyproject.toml`. Run `uv lock` from the recipe root:

```bash
cd contrib/python/my-recipe
uv lock
```

> **Note:** a stale `uv.lock` can cause both `python-dependency-policy.yml`
> and `python-tests.yml` to fail. Fix this first if both checks are red.

## pyproject.toml has no sibling uv.lock

**Workflow:** [`python-dependency-policy.yml`](../../.github/workflows/python-dependency-policy.yml)

CI output contains: `has no sibling uv.lock`

Every `pyproject.toml` with a `[project]` table or `[tool.uv]`
section needs a `uv.lock` next to it. Run `uv lock` in that
directory:

```bash
cd contrib/python/my-recipe
uv lock
```

## Lockfile references a non-PyPI URL

**Workflow:** [`python-dependency-policy.yml`](../../.github/workflows/python-dependency-policy.yml)

CI output contains: `contains non-PyPI registry URLs`

Every entry in `uv.lock` must resolve to `pypi.org` or
`files.pythonhosted.org`. Internal registries, GitHub Packages,
Artifactory, and similar sources are not allowed. Replace the
dependency with a PyPI-published package.

## VCS dependency in uv.lock

**Workflow:** [`python-dependency-policy.yml`](../../.github/workflows/python-dependency-policy.yml)

CI output contains: `contains git/VCS dependencies`

`source = { git = "..." }` entries are not allowed. Git dependencies
are non-reproducible and skip the package registry's security
verification. Publish the package to PyPI, or use a PyPI equivalent.

## Local path dependency in uv.lock

**Workflow:** [`python-dependency-policy.yml`](../../.github/workflows/python-dependency-policy.yml)

CI output contains: `contains local path dependencies`

`path`, `editable`, or `directory` sources only exist on the
committer's machine and cannot be resolved in other environments.
Exception: uv's self-referential `editable = "."` entry for the
workspace root package is allowed. Remove all other local path
sources.

## Missing package hash in uv.lock

**Workflow:** [`python-dependency-policy.yml`](../../.github/workflows/python-dependency-policy.yml)

CI output contains: `All distributions must have sha256 hashes`

Every distribution must include a `sha256` hash. Missing hashes
usually mean the lockfile was hand-edited or generated with a
different tool. Regenerate cleanly:

```bash
cd contrib/python/my-recipe
rm uv.lock
uv lock
```

## Ruff format or check failed

**Workflow:** [`python-format.yml`](../../.github/workflows/python-format.yml)

CI output contains: `Would reformat` (format failure) or a Ruff rule ID such as `E`, `W`, `F`, `I` followed by a description (lint failure)

Auto-fix most issues:

```bash
uv run ruff format <recipe-path>
uv run ruff check --fix <recipe-path>
```

Some issues require manual fixing — Ruff reports which ones and cites
the rule ID.

---

**The following will not block your PR.** Fix them when convenient.

### Non-blocking notices

- **Hardcoded model name** — replace with `os.getenv("MODEL_NAME")`.
  Run `extract-python-environment-variables`.
- **`GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` / `MODEL_NAME`
  missing from `.env.example`** — add them if your recipe uses them.

## Something else

If your error isn't listed here, check the workflow log for the exact
error message and the file it references. Then either:

- Search this page for keywords from the error message.
- Open a GitHub issue at
  [github.com/google/adk-samples/issues](https://github.com/google/adk-samples/issues)
  with the workflow name, error message, and a link to the failed run.
  Use this template:

  ```
  **Recipe path:** contrib/python/my-recipe
  **Workflow:** python-validate-recipe.yml
  **Error:** (paste the relevant CI log lines here)
  **Failed run:** (link to the GitHub Actions run)
  ```

---

_Last updated: July 2025_

← [Checklist](../recipe-checklist.md) · [Handbook](./README.md)
