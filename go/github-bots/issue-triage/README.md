# Issue Triage Bot

An autonomous [ADK Go](https://github.com/google/adk-go) agent that triages open
GitHub issues. For each untriaged issue it:

1. **Sets the issue type** — `Bug`, `Feature`, or `Task`.
2. **Applies one categorization label** from a configurable allowlist
   (`bug`, `enhancement`, `documentation`, `question` by default).

An issue is considered **untriaged** when it has no issue type and/or none of
the allowlisted categorization labels. Several invariants are enforced **in Go**,
not merely requested in the prompt: the type and label must be in the allowlist;
the agent may only mutate issues it legitimately targeted (the single `-issue`,
or those returned by `list_untriaged_issues`); and it may only fill a field that
was actually missing, so an already-set type or label is never overwritten.

## What it demonstrates

- An `llmagent.New` agent driven by typed `functiontool.New[Args, Result]` tools.
- Running headlessly with a `runner.Runner` + in-memory session, consuming the
  streaming `iter.Seq2[*session.Event, error]` response.
- Calling the GitHub REST API (`go-github`) and GraphQL API (a raw POST through
  the same authenticated client) from inside tools.
- A clean split between pure, table-tested decision logic (`triage.go`) and
  side-effecting I/O (`github.go`): deterministic facts in code, fuzzy
  classification in the model.

## The agent loop

If you are new to ADK, this is the core flow (`main.go`):

1. `llmagent.New` is given the model, the rendered instruction, and the tools.
   ADK reflects over each tool's argument struct (via `functiontool.New`) to
   build the JSON schema the model sees — **the Go arg struct is the tool's
   input contract**.
2. `runner.New` binds the agent to a `SessionService`; `runner.Run(...)` returns
   an `iter.Seq2[*session.Event, error]` (a Go 1.23 range-over-func) yielding one
   streamed event or an error per iteration.
3. On each turn the model reads the prompt + tool schemas, may emit a tool call,
   the runner executes the matching Go handler, feeds the result back, and loops
   until the model stops calling tools and returns text.
4. We consume that stream headlessly, keep the last text as the summary, and
   return a non-nil error if any event carried one (so CI fails loudly).

Validation failures (e.g. a disallowed label) are returned to the model as a
result with `status: "error"` and a **nil Go error** so it can self-correct;
real I/O failures return a Go `error`. `OnToolErrorCallbacks` returns
`(nil, nil)` — "observe only" — to log failures that are otherwise invisible.

> **Why GraphQL for reads but REST for writes?** Not an ADK convention — GitHub's
> issue *type* is not exposed by the REST API in `go-github` v66, so reads use
> GraphQL (`issueType { name }`) and the type write is a raw `PATCH`. Labels use
> the regular REST endpoint.

## Running locally

Requires **Go 1.25+** (see `go.mod`). Copy `.env.example` to `.env` and fill it
in (or export the variables), then:

```bash
# Dry-run a single issue (no writes; logs intended actions).
go run . -dry-run -issue 123

# Dry-run a sweep of the backlog.
go run . -dry-run

# Act for real (omit -dry-run).
go run . -issue 123
```

> **Dry-run is not offline.** `-dry-run` still reads GitHub and still calls the
> model; it only suppresses writes, logging `would …` instead.

## Configuration

| Variable / flag | Default | Description |
| --- | --- | --- |
| `GITHUB_TOKEN` | — (required) | Token with `issues: write`. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini API key (or use Vertex AI). |
| `OWNER` | — (required) | Repository owner. |
| `REPO` | — (required) | Repository name. |
| `LLM_MODEL_NAME` | `gemini-flash-latest` | Model to use. |
| `ALLOWED_LABELS` | `bug,enhancement,documentation,question` | Categorization label allowlist. |
| `ISSUE_COUNT` | `3` | Max issues per scheduled sweep (newest first). |
| `FRESHNESS_WINDOW_DAYS` | `0` (off) | Restrict the sweep to issues created within N days. |
| `ISSUE_TIMEOUT` | `5m` | Bounds a single agent run. |
| `SWEEP_TIMEOUT` | `15m` | Bounds the whole run, so N issues cannot multiply into N x `ISSUE_TIMEOUT` and overrun the job timeout. Must be at least `ISSUE_TIMEOUT`. |
| `-dry-run` / `DRY_RUN` | `false` | Log intended actions without mutating. |
| `-issue` | `0` | Triage only this issue (0 = sweep). |

Instead of an API key you can use Vertex AI via Application Default Credentials
(`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`).

## Running it against another repository (GitHub Actions)

This bot is designed to run on a *target* repo (e.g. `google/adk-go`) without
copying the code there. It ships a **reusable workflow**
(`.github/workflows/go-issue-triage.yml` in this repo, `on: workflow_call`); the
target repo adds a small **caller** workflow:

```yaml
# in the target repo, e.g. google/adk-go: .github/workflows/triage-bot.yml
on:
  issues: { types: [opened] }
  schedule: [{ cron: '0 */6 * * *' }]
  workflow_dispatch:
    inputs:
      issue:   { type: string,  required: false }
      dry_run: { type: boolean, required: false, default: true }
jobs:
  triage:
    if: github.repository == 'google/adk-go'
    permissions: { issues: write, contents: read }
    uses: google/adk-samples/.github/workflows/go-issue-triage.yml@<commit-sha>
    with:
      issue_number: ${{ github.event.issue.number || inputs.issue }}
      dry_run: ${{ github.event_name == 'workflow_dispatch' && inputs.dry_run || false }}
      # Pin the bot CODE to the same commit as the workflow above. Omitting this
      # runs adk-samples@main, so a pinned `uses:` would still execute unpinned
      # code — set both to the same <commit-sha> for a reproducible run.
      samples_ref: <commit-sha>
    secrets:
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

Because a reusable workflow runs in the **caller's** context, the built-in
`GITHUB_TOKEN` (scoped by the caller's `permissions:`) writes to the target repo
— no PAT or GitHub App required — and `OWNER`/`REPO` resolve to the caller.

## Tests

Pure logic is table-driven (`triage_test.go`); the GitHub client is exercised
with `httptest` (`github_test.go`, incl. GraphQL pagination and PR/NOT_FOUND
handling); the tool layer's allowlist/authorization gates are verified to reject
bad input without any HTTP call (`tools_test.go`).

```bash
go test ./...
```

## Notes

- **Issue types** must be enabled at the organization level (they are for the
  `google` org: Bug/Feature/Task). Setting a type — and adding a label — requires
  a token with **push access**; without it GitHub returns success but silently
  drops the change. The bot reads back each write and **fails the run** if the
  type/label was not actually applied, so a permissions gap surfaces loudly
  instead of passing silently. In the reusable workflow, `issues: write` alone is
  not push access: if type/label changes are being dropped, grant the caller job
  `contents: write` (or run the bot with a token that has push access).
- **Component labels / owner assignment** from the original Python
  `adk_triaging_agent` are intentionally omitted; both are natural extensions.
