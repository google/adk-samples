# GitHub Workflows

This repository contains the following GitHub Actions workflows:

| Workflow | File | Purpose |
|----------|------|---------|
| [💬 Antigravity CLI](#-antigravity-cli) | `agy-cli.yml` | General-purpose AI coding assistant invoked via `@agy` |
| [🏷️ Antigravity Automated Issue Triage](#️-antigravity-automated-issue-triage) | `agy-issue-automated-triage.yml` | Triages a single issue with labels on open/reopen or on demand |
| [📋 Antigravity Scheduled Issue Triage](#-antigravity-scheduled-issue-triage) | `agy-issue-scheduled-triage.yml` | Batch-triages all unlabeled issues on a schedule |
| [🧐 Antigravity Pull Request Review](#-antigravity-pull-request-review) | `agy-pr-review.yml` | AI code review posted directly to GitHub PRs |
| [✅ Checks](#-checks) | `ruff-checks.yml` | Ruff linting and formatting suggestions on Python PRs |
| [Dependabot Auto-Merge](#dependabot-auto-merge) | `dependabot-auto-merge.yml` | Automatically approves and merges Dependabot PRs that pass checks |
| [Agent Template Tests](#agent-template-tests) | `test_templated_agent.yaml` | Lints and tests changed agent templates against deployment targets |

---

## Antigravity Prerequisites

The Antigravity workflows (`agy-*.yml`) require at least one of the following to be configured in the **PROD** environment in the repository settings:

| Setting | Type | Description |
|---------|------|-------------|
| `GEMINI_API_KEY` | Secret | API key for Gemini (direct API access) |
| `GOOGLE_CLOUD_PROJECT` | Variable | GCP project ID for Vertex AI access |
| `GOOGLE_CLOUD_LOCATION` | Variable | GCP region (e.g. `us-central1`) |
| `GOOGLE_GENAI_USE_VERTEXAI` | Variable | Set to `true` to use Vertex AI instead of direct Gemini API |
| `GOOGLE_GENAI_USE_GCA` | Variable | Set to `true` to use Gemini Code Assist instead of direct API |
| `GCP_WIF_PROVIDER` | Variable | Workload Identity Federation provider (required for Vertex AI auth) |
| `SERVICE_ACCOUNT_EMAIL` | Variable | GCP service account email (required for Vertex AI auth) |
| `APP_ID` | Variable | GitHub App ID (optional, for elevated permissions) |
| `APP_PRIVATE_KEY` | Secret | GitHub App private key (required if `APP_ID` is set) |

---

## Workflows

### 💬 Antigravity CLI (`agy-cli.yml`)

A general-purpose AI coding assistant that responds to `@agy` mentions in issues and pull requests. It can investigate problems, make code changes, commit and push fixes, create PRs, and answer questions about the codebase.

#### Automatic triggers

| Event | Condition |
|-------|-----------|
| Issue opened | Body contains `@agy` (but not `@agy /review` or `@agy /triage`) |
| Issue comment created | Comment contains `@agy` (but not `@agy /review` or `@agy /triage`) |
| PR review submitted | Review body contains `@agy` (but not `@agy /review` or `@agy /triage`) |
| PR review comment created | Comment contains `@agy` (but not `@agy /review` or `@agy /triage`) |

Only users with `OWNER`, `MEMBER`, or `COLLABORATOR` association can invoke the bot. Bot senders are explicitly blocked to prevent infinite loops.

#### Usage examples

Invoke by mentioning `@agy` followed by your request in any issue or PR comment:

```
@agy Can you investigate why the integration tests are failing and fix the root cause?
```

```
@agy Please refactor the `DataProcessor` class to use async/await instead of callbacks.
```

```
@agy Answer this question: what does the `run_pipeline` function do and how does it handle errors?
```

#### Manual trigger (`workflow_dispatch`)

Can be triggered manually from the **Actions** tab with no inputs required. Useful for testing the workflow itself.

---

### 🏷️ Antigravity Automated Issue Triage (`agy-issue-automated-triage.yml`)

Automatically triages a single issue by analyzing its title and body and applying appropriate labels (following `kind/*` and `priority/*` patterns). Also removes the `status/needs-triage` label if present.

#### Automatic triggers

| Event | Condition |
|-------|-----------|
| Issue opened | Sender is not a bot |
| Issue reopened | Sender is not a bot |
| Issue comment created | Comment contains `@agy /triage`, sender is not a bot |

Only users with `OWNER`, `MEMBER`, or `COLLABORATOR` association can invoke via comment.

#### Usage examples

To manually re-triage an issue, leave a comment on it:

```
@agy /triage
```

#### Manual trigger (`workflow_dispatch`)

Can be triggered manually from the **Actions** tab with the following input:

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `issue_number` | number | Yes | The number of the issue to triage |

**Steps to run manually:**
1. Go to the **Actions** tab in the repository.
2. Select **Antigravity Automated Issue Triage** from the left sidebar.
3. Click **Run workflow**.
4. Enter the issue number (e.g. `42`).
5. Click **Run workflow**.

---

### 📋 Antigravity Scheduled Issue Triage (`agy-issue-scheduled-triage.yml`)

Runs on a schedule to batch-triage all open issues that either have no labels or have the `status/needs-triage` label. Fetches up to 100 issues per query and processes them in a single AI session.

#### Automatic triggers

| Event | Condition |
|-------|-----------|
| Schedule | Every hour (`0 * * * *`) |

#### Manual trigger (`workflow_dispatch`)

Can be triggered manually from the **Actions** tab with no inputs required. Useful for running an immediate triage sweep outside of the hourly schedule.

**Steps to run manually:**
1. Go to the **Actions** tab in the repository.
2. Select **Antigravity Scheduled Issue Triage** from the left sidebar.
3. Click **Run workflow**.
4. Click **Run workflow** to confirm.

---

### 🧐 Antigravity Pull Request Review (`agy-pr-review.yml`)

Performs an automated AI code review on a pull request, posting inline comments and a summary review directly on GitHub using the GitHub MCP tools. Reviews focus on correctness, efficiency, maintainability, and security. Comments are severity-labeled (🔴 critical, 🟠 high, 🟡 medium, 🟢 low).

#### Automatic triggers

| Event | Condition |
|-------|-----------|
| PR opened | Author has `OWNER`, `MEMBER`, or `COLLABORATOR` association, sender is not a bot |
| PR reopened | Author has `OWNER`, `MEMBER`, or `COLLABORATOR` association, sender is not a bot |
| Issue comment on a PR | Comment contains `@agy /review`, sender is not a bot |
| PR review comment | Comment contains `@agy /review`, sender is not a bot |
| PR review submitted | Review body contains `@agy /review`, sender is not a bot |

Only users with `OWNER`, `MEMBER`, or `COLLABORATOR` association can invoke via comment. Bot senders are explicitly blocked to prevent infinite loops.

#### Usage examples

To request a review on any open PR, leave a comment on it:

```
@agy /review
```

You can also provide specific focus areas after the command:

```
@agy /review focus on security and input validation
```

```
@agy /review check for breaking changes and performance regressions
```

```
@agy /review look at error handling in the new API endpoints
```

#### Manual trigger (`workflow_dispatch`)

Can be triggered manually from the **Actions** tab with the following input:

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `pr_number` | number | Yes | The number of the PR to review |

**Steps to run manually:**
1. Go to the **Actions** tab in the repository.
2. Select **Antigravity Pull Request Review** from the left sidebar.
3. Click **Run workflow**.
4. Enter the PR number (e.g. `123`).
5. Click **Run workflow**.

---

## ✅ Checks

**File:** `ruff-checks.yml`

Runs [Ruff](https://docs.astral.sh/ruff/) linting and formatting on Python agent code when a PR touches `python/agents/**` or `pyproject.toml`. Uses [Reviewdog](https://github.com/reviewdog/reviewdog) to post inline code suggestions directly on the PR diff, so authors can apply fixes with a single click.

#### Triggers

| Event | Condition |
|-------|-----------|
| PR opened/updated targeting `main` | Changes in `python/agents/**` or `pyproject.toml` |

No manual trigger. No configuration required beyond the default `GITHUB_TOKEN`.

---

## Dependabot Auto-Merge

**File:** `dependabot-auto-merge.yml`

Automatically approves and merges Dependabot PRs, but only if:
- All CI checks pass
- The changes are in `python/agents/**` directories
- Each changed agent directory uses the `agent-starter-pack` testing framework (has `[tool.agent-starter-pack]` in its `pyproject.toml`)
- The PR is not in a conflicting state (triggers a rebase request if so)

This avoids auto-merging dependency updates for agents that lack automated test coverage.

#### Triggers

| Event | Condition |
|-------|-----------|
| PR opened/synchronized/reopened | Actor is `dependabot[bot]` |
| `checks` workflow completes successfully | Checks if the completed run's PR is from Dependabot |

No manual trigger. No configuration required beyond the default `GITHUB_TOKEN`.

---

## Agent Template Tests

**File:** `test_templated_agent.yaml`

Discovers which agent directories under `python/agents/**` have changed in a PR, then runs lint and/or integration tests for each changed agent using the `agent-starter-pack` framework. Tests run in a Docker container against GCP using Workload Identity Federation.

Agents are skipped if:
- They contain a `.testignore` file
- Their `pyproject.toml` does not contain a `[tool.agent-starter-pack]` section

#### Triggers

| Event | Condition |
|-------|-----------|
| PR opened/updated | Always — discovers changed agents dynamically |

No manual trigger.

#### GCP configuration

This workflow authenticates to GCP using a hardcoded Workload Identity Federation provider and project (`adk-devops`). No repository-level secrets or variables are required.
