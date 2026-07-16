# Stale-Issue Auditor Bot

An autonomous [ADK Go](https://github.com/google/adk-go) agent that audits open
GitHub issues for staleness. Unlike a timestamp-only "stale bot", it
reconstructs each issue's full conversation history and uses an LLM to tell the
difference between a maintainer **asking the author a question** (a stale
candidate) and a maintainer **posting a status update** (still active).

## What it demonstrates

- An `llmagent.New` agent driven by typed `functiontool.New[Args, Result]` tools.
- Running headlessly with a `runner.Runner` + in-memory session, consuming the
  streaming `iter.Seq2[*session.Event, error]` response.
- Calling the GitHub REST (`go-github`) and GraphQL APIs from inside tools.
- **Bounded concurrency** with `errgroup`, deterministic decisions
  (`Temperature: 0`), and a clean split between **pure, unit-tested logic**
  (`state.go`) and **side-effecting I/O** (`github.go`).

## How it works

For each candidate issue the agent:

1. Calls `get_issue_state`, which issues one GraphQL query (comments,
   description edits, title renames, reopen/label events), replays the history to
   find the **last human actor**, and computes staleness.
2. Follows a decision tree (`prompt_instruction.txt`):
   - **Author/other replied** → remove the stale label (active again); if they
     edited the description silently, alert maintainers.
   - **Maintainer asked a question** and the stale threshold passed → mark stale
     (warning comment + label).
   - **Stale long enough** → close as *not planned*.
   - **Maintainer status update / internal discussion** → no action.

Mutations are ordered for safe re-runs (label before comment; close before
comment) and the bot recognizes its own prior comments to avoid spam.

## Running locally

Requires **Go 1.25+** (see `go.mod`). Copy `.env.example` to `.env` and fill it
in (set `MAINTAINERS` — without it the bot will never mark issues stale), then:

```bash
# Dry-run the whole backlog (no writes; logs intended actions).
go run . -dry-run

# Dry-run a single issue.
go run . -dry-run -issue 123

# Act for real (omit -dry-run).
go run .
```

> **Dry-run is not offline.** It still reads GitHub and calls the model; it only
> suppresses writes.

## Configuration

| Variable / flag | Default | Description |
| --- | --- | --- |
| `GITHUB_TOKEN` | — (required) | Token with `issues: write`. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini API key (or use Vertex AI). |
| `MAINTAINERS` | — | Comma-separated maintainer logins (the token can't list collaborators). Without it, no comment counts as maintainer activity, so nothing is ever marked stale. |
| `OWNER` | — (required) | Repository owner. |
| `REPO` | — (required) | Repository name. |
| `LLM_MODEL_NAME` | `gemini-flash-latest` | Model to use. |
| `STALE_HOURS_THRESHOLD` | `336` (14d) | Time waiting on the author before warning. |
| `CLOSE_HOURS_AFTER_STALE_THRESHOLD` | `168` (7d) | Time stale before closing. |
| `STALE_LABEL_NAME` | `stale` | Label applied when marking stale. |
| `REQUEST_CLARIFICATION_LABEL` | `request clarification` | Label that flags "waiting on author". |
| `CONCURRENCY_LIMIT` | `3` | Max issues audited in parallel. |
| `ISSUE_TIMEOUT` | `5m` | Bounds a single issue's audit. |
| `-dry-run` / `DRY_RUN` | `false` | Log intended actions without mutating. |
| `-issue` | `0` | Audit only this issue (0 = sweep). |

Instead of an API key you can use Vertex AI via Application Default Credentials
(`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`).

## Running it against another repository (GitHub Actions)

This bot ships a **reusable workflow** (`.github/workflows/go-stale-issues.yml`
in this repo, `on: workflow_call`); the target repo adds a thin **caller**
workflow:

```yaml
# in the target repo, e.g. google/adk-go: .github/workflows/stale-bot.yml
on:
  schedule: [{ cron: '0 7 * * *' }]   # daily
  workflow_dispatch:
    inputs:
      issue:   { type: string,  required: false }
      dry_run: { type: boolean, required: false, default: true }
jobs:
  stale:
    if: github.repository == 'google/adk-go'
    permissions: { issues: write, contents: read }
    uses: google/adk-samples/.github/workflows/go-stale-issues.yml@<commit-sha>
    with:
      issue_number: ${{ inputs.issue }}
      dry_run: ${{ github.event_name == 'workflow_dispatch' && inputs.dry_run || false }}
      maintainers: 'login1,login2,login3'
      # Pin the bot CODE to the same commit as the workflow above. Omitting this
      # runs adk-samples@main, so a pinned `uses:` would still execute unpinned
      # code — set both to the same <commit-sha> for a reproducible run.
      samples_ref: <commit-sha>
    secrets:
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

A reusable workflow runs in the **caller's** context, so the built-in
`GITHUB_TOKEN` (scoped by the caller's `permissions:`) writes to the target repo
— no PAT or GitHub App required.

## Tests

Pure decision logic is table-driven (`state_test.go`); the GitHub client is
exercised with `httptest` (`github_test.go`); config and prompt rendering are
unit-tested.

```bash
go test ./...
```

## Notes

- The `MAINTAINERS` list must be supplied explicitly — the repo-scoped
  `GITHUB_TOKEN` cannot read collaborators/teams at runtime.
- Timeouts are policy: the defaults (14d to stale, 7d to close) suit OSS
  volunteer cadence; tune via env.
