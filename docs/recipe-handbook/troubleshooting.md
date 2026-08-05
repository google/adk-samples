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
- [Required file or directory missing](#required-file-or-directory-missing)
- [Recipe is in the wrong folder](#recipe-is-in-the-wrong-folder)
- [Changes inside a retired folder](#changes-inside-a-retired-folder)

**README.md**
- [README.md is missing or empty](#readmemd-is-missing-or-empty)
- [README.md contains TODO placeholders](#readmemd-contains-todo-placeholders)
- [README.md is too short](#readmemd-is-too-short)
- [README.md is missing a setup section](#readmemd-is-missing-a-setup-section)
- [README.md is missing a run section or code block](#readmemd-is-missing-a-run-section-or-code-block)

**pyproject.toml**
- [pyproject.toml has a local ruff configuration](#pyprojecttoml-has-a-local-ruff-configuration)
- [Standalone Ruff config file](#standalone-ruff-config-file)
- [Project name doesn't match the required name](#project-name-doesnt-match-the-required-name)
- [Project description doesn't match manifest](#project-description-doesnt-match-manifest)
- [requires-python below 3.11](#requires-python-below-311)
- [pyproject.toml has no sibling uv.lock](#pyprojecttoml-has-no-sibling-uvlock)
- [Missing [[tool.uv.index]] block](#missing-tooluvindex-block)

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
- [CI infrastructure failure](#ci-infrastructure-failure)
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

## Required file or directory missing

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `Required file '<name>' is missing` or
`Required directory '<name>/' is missing`

The set of required entries is the **union** of three rules in
[`.github/policy.yml`](../../.github/policy.yml). The CI message names
which one applied to your recipe:

| Rule | Applies to | Entries |
| --- | --- | --- |
| `always` | every recipe | `README.md` |
| `by_root.core` | anything under `core/` | `AGENTS.md` |
| `by_root.skills` | anything under `skills/` | `SKILL.md`, `EVAL.yaml`, `scripts/` |
| `by_language.python` | `manifest.language: python` | `pyproject.toml`, `uv.lock`, `.env.example`, `tests/test_runnability.py` |

`manifest.yaml` is required for every recipe and reported separately.

Note that language requirements come from **`manifest.language`**, not
from the folder path. A vertical skill at `skills/retail/product-search`
picks up the Python list because its manifest says so — the middle folder
is a vertical, not a language.

Most of these have a generator:

| Missing | Fix |
| --- | --- |
| `manifest.yaml` | `generate-manifest` (AI skill) |
| `.env.example` | `extract-python-environment-variables` (AI skill) |
| `tests/test_runnability.py` | `generate-python-runnability-test` (AI skill) |
| `uv.lock` | `uv lock` in the recipe directory |
| `pyproject.toml` | `align-recipe-pyproject` (AI skill) |
| everything at once | `scaffold-python-recipe` (AI skill) |

**Directory looks present but CI says it's missing?** Git cannot commit an
empty directory. Add a placeholder file and commit that:

```bash
touch scripts/.gitkeep && git add scripts/.gitkeep
```

## Recipe is in the wrong folder

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `sits directly under` or `is nested too deeply`

Every recipe under `skills/` must live at
`skills/<vertical>/<solution>/`. The vertical (`retail/`, `hr/`,
`finance/`) is mandatory — it surfaces ownership and lets a team see its
whole surface at a glance. A solution dropped directly under `skills/`
has no owning vertical and is rejected.

```
skills/retail/product-search/manifest.yaml    valid
skills/product-search/manifest.yaml           too shallow — no vertical
skills/retail/product-search/x/manifest.yaml  too deep
```

Move the directory to the path shown in the error, then re-run
`uv run validate placement`.

Remember that `[project].name` in `pyproject.toml` is
`<vertical>-<solution>` for a vertical skill (`retail-product-search`),
not the folder basename — see
[Project name doesn't match the required name](#project-name-doesnt-match-the-required-name).

## Changes inside a retired folder

**Workflow:** [`validate-recipe-structure.yml`](../../.github/workflows/validate-recipe-structure.yml)

CI output contains: `is in a retired folder and no longer accepts changes`

Recipes used to live at `<language>/agents/<recipe>` in the repo root (for
example `python/agents/academic-research`). Those roots are closed. The
retired roots are listed under `frozen_paths` in
[`.github/policy.yml`](../../.github/policy.yml).

Move the whole recipe to `contrib/<language>/<recipe>` and make the change
there. The error names the destination it expects.

Deletions and renames are exempt, so migrating a recipe *out* of a retired
folder is never blocked by this check — only adding to or editing one in
place is.

After moving, the recipe has to meet the current contribution
requirements, which the retired copy predates: see
[the checklist](../recipe-checklist.md) and
[Required file or directory missing](#required-file-or-directory-missing).

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

## Project name doesn't match the required name

**Workflow:** [`python-validate-recipe.yml`](../../.github/workflows/python-validate-recipe.yml)

CI output contains: `[project].name` and `does not match the required name`

Set `[project].name` in `pyproject.toml` to the name CI reports as
required. It is derived from where the recipe lives:

- `core/` and `contrib/` — the recipe folder basename. A recipe at
  `contrib/python/my-recipe` needs `name = "my-recipe"`.
- `skills/` — `<vertical>-<solution>`, because `skills/` interposes a
  mandatory vertical namespace. A skill at
  `skills/retail/product-search` needs
  `name = "retail-product-search"`, not `product-search`.

```toml
# contrib/python/my-recipe
name = "my-recipe"

# skills/retail/product-search
name = "retail-product-search"
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

## Missing [[tool.uv.index]] block

**Workflow:** [`python-validate-recipe.yml`](../../.github/workflows/python-validate-recipe.yml)

CI output contains: `Missing required [[tool.uv.index]] block` or
`has default=true but url=`

Every recipe must declare public PyPI as its default index. Add this to
the recipe's `pyproject.toml`:

```toml
[[tool.uv.index]]
url = "https://pypi.org/simple/"
default = true
```

Note the **double** brackets: `[[tool.uv.index]]` is an array of tables.
A single-bracket `[tool.uv.index]` is a different TOML construct and uv
will not accept it.

Why it is required: on a Google corp workstation a system-wide
`/etc/uv/uv.toml` redirects package resolution to an authenticated proxy.
uv concatenates project-level indexes ahead of system-level ones, so
declaring PyPI here puts it first and `uv sync` works without that auth.

`align-recipe-pyproject` (AI skill) adds the block for you.

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

## CI infrastructure failure

CI output contains: `[ci-fault]` or `CI tooling failure in`

**This is not caused by your changes.** One of the repo's own checker
scripts crashed, or the CI environment failed (network, a missing
dependency, an unhandled file encoding).

You will see this instead of a normal error when the failure is ours
rather than yours. It is annotated against the workflow, not against
your files, precisely so you do not go hunting for a bug in your PR.

What to do:

1. Re-run the failed job. Genuinely transient failures (network, registry
   timeouts) clear on a retry.
2. If it fails again, it needs a repo maintainer. Open an issue with the
   workflow name, the `Detail:` line from the output, and a link to the
   run. Do not change your recipe to work around it.

If the crash was triggered by something unusual in your recipe — an
unusual file encoding, a hand-edited `uv.lock` — say so in the issue.
The checker should report that as a clear error rather than crash, so
it is a bug in the checker either way.

## Non-blocking notices

**The following will not block your PR.** Fix them when convenient.

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
