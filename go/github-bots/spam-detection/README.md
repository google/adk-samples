# Spam Detection Bot

An autonomous [ADK Go](https://github.com/google/adk-go) agent that moderates a
GitHub repository's issues for spam. It audits open issues for SEO spam,
unsolicited promotion of third-party products or sites, and other off-topic
solicitation. When the model judges an issue's content to be spam it:

1. **Applies the spam label** (`spam` by default), and
2. **Posts one comment** alerting the maintainers, with a short reason.

Several invariants are enforced **in Go**, not merely requested in the prompt:
the bot only reviews non-maintainer, non-bot content; it never re-processes an
issue it has already labeled or alerted (idempotency); and the model may only
flag the single issue its session is scoped to, so injected instructions in
(untrusted) issue or comment text cannot redirect it to another issue.

## What it demonstrates

- An `llmagent.New` agent driven by a single typed `functiontool.New[Args, Result]`
  tool, run **once per issue in its own isolated session** with bounded
  concurrency (`errgroup`) — i.e. code orchestrates the loop and the model only
  classifies.
- A clean split between deterministic work done **in code** and the fuzzy
  judgment delegated to the **model**: the bot fetches the issue, filters out
  maintainer/bot/already-handled content, truncates long text, and annotates each
  author with their GitHub **author association** (a spam-likelihood prior) in
  `spam.go`, then asks the model only "is this spam?" — guided by that signal and
  a few worked examples in the prompt.
- The **zero-waste** optimization from the original Python sample: if nothing
  reviewable remains after filtering (or the issue was already handled), the
  model is never invoked.
- Calling the GitHub REST API (`go-github`) and GraphQL API (a raw POST through
  the same authenticated client) from code.

## The agent loop

If you are new to ADK, this is the core flow (`main.go`):

1. Code selects the candidate issues (a sweep via the Search API, or a single
   `-issue`).
2. For each issue, in its own goroutine (bounded by `CONCURRENCY_LIMIT`),
   `reviewIssue` fetches the issue + comments, runs the idempotency and filtering
   logic, and assembles the reviewable text. Issues with nothing to review are
   skipped without a model call.
3. The agent runs in a fresh, **issue-scoped** session. The prompt carries the
   issue number and the assembled content (clearly fenced and marked untrusted);
   `runner.Run(...)` returns an `iter.Seq2[*session.Event, error]` that yields one
   streamed event or an error per iteration.
4. The model either calls `flag_issue_as_spam(issue_number, detection_reason)`
   or replies "No spam detected." `authorizeIssue` rejects any `issue_number`
   other than the one the session is scoped to.

A rejected tool call (wrong issue) is returned to the model as a result with
`status: "error"` and a **nil Go error**; real I/O failures return a Go `error`,
are recorded, and make the process exit non-zero so scheduled/CI runs fail
loudly.

> **Why embed the content in the prompt instead of a retrieval tool?** Because
> all the deterministic pre-processing (filtering, truncation, and the
> idempotency check that lets us skip the model entirely) happens in code
> before the model runs. Putting the finished, untrusted-marked text in the
> per-issue prompt keeps the model's only job — and its only tool — the spam
> decision itself.

## Running locally

Requires **Go 1.25+** (see `go.mod`). Copy `.env.example` to `.env` and fill it
in (or export the variables), then:

```bash
# Dry-run a single issue (no writes; logs intended actions).
go run . -dry-run -issue 123

# Dry-run a sweep of recent issues.
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
| `SPAM_LABEL_NAME` | `spam` | Label applied to flagged issues (must already exist). |
| `MAINTAINERS` | (empty) | Comma-separated logins whose comments are trusted and never reviewed. |
| `ISSUE_COUNT` | `3` | Max issues per scheduled sweep (most-recently-updated first). |
| `CONCURRENCY_LIMIT` | `3` | How many issues to review in parallel. |
| `FRESHNESS_WINDOW_DAYS` | `0` (off) | Restrict the sweep to issues updated within N days. |
| `ISSUE_TIMEOUT` | `5m` | Bounds a single issue review. |
| `-dry-run` / `DRY_RUN` | `false` | Log intended actions without mutating. |
| `-issue` | `0` | Review only this issue (0 = sweep). |

Instead of an API key you can use Vertex AI via Application Default Credentials
(`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`).

> **`MAINTAINERS` is optional but recommended.** The built-in Actions
> `GITHUB_TOKEN` cannot list a repo's collaborators, so trusted logins are
> supplied explicitly. With an empty set the bot still works — it just also
> reviews maintainers' own comments (wasting a few tokens and risking a
> false positive); it never misses spam because of it.

## Running it against another repository (GitHub Actions)

This bot is designed to run on a *target* repo (e.g. `google/adk-go`) without
copying the code there. It ships a **reusable workflow**
(`.github/workflows/go-spam-detection.yml` in this repo, `on: workflow_call`);
the target repo adds a small **caller** workflow:

```yaml
# in the target repo, e.g. google/adk-go: .github/workflows/spam-bot.yml
on:
  issues: { types: [opened] }
  issue_comment: { types: [created] }
  schedule: [{ cron: '0 */6 * * *' }]
  workflow_dispatch:
    inputs:
      issue:   { type: string,  required: false }
      dry_run: { type: boolean, required: false, default: true }
jobs:
  spam:
    if: github.repository == 'google/adk-go'
    permissions: { issues: write, contents: read }
    uses: google/adk-samples/.github/workflows/go-spam-detection.yml@<commit-sha>
    with:
      issue_number: ${{ github.event.issue.number || inputs.issue }}
      dry_run: ${{ github.event_name == 'workflow_dispatch' && inputs.dry_run || false }}
      maintainers: 'octocat,hubot'
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

Pure logic is table-driven (`spam_test.go`: filtering, case-insensitive
maintainer/identity matching, idempotency, truncation, suspect-text assembly,
prompt-injection-resistant signature detection); the GitHub client is exercised
with `httptest` (`github_test.go`, incl. PR exclusion, repo-vs-issue NOT_FOUND
handling, and comment-before-label ordering); the tool layer's issue-scope
authorization, within-run idempotency, and dry-run gates are verified to reject
bad input without any HTTP call (`tools_test.go`).

```bash
go test ./...
```

## Differences from the Python sample

This is adapted from the Python `adk_issue_monitoring_agent`. The behavior
differs in a few deliberate ways:

- **Scan strategy.** Python had an uncapped daily sweep (everything updated in
  the last 24h, via `since`) plus an `INITIAL_FULL_SCAN` of every open issue.
  This bot caps the sweep at `ISSUE_COUNT` (most-recently-updated first) and
  treats it as a **backstop** to the workflow's `issues`/`issue_comment` event
  triggers (which catch new spam in real time). If you run it on a schedule only,
  raise `ISSUE_COUNT` so the backstop covers your volume.
- **Title review.** The issue title is reviewed in addition to the body (Python
  reviewed only the body), so spam titles are caught.
- **Code blocks kept.** Python stripped fenced code blocks before review; this
  bot keeps them (bounded by truncation) so spam can't hide inside a ``` fence.
- **Alert comment.** A plain "Maintainers, please review." line rather than
  Python's literal `@maintainers` mention (which only pings if such a team/user
  exists).
- **Concurrency.** Bounded by `errgroup` (`CONCURRENCY_LIMIT`) rather than
  Python's fixed chunk size plus an inter-batch sleep.
- **Idempotency.** The spam **label** is the primary guard; within a run a second
  flag of the same issue is a no-op in code; the bot's own alert comment is a
  best-effort secondary signal that can be missed on threads with more comments
  than the fetch window after the alert (only causing a re-alert if the label was
  also removed).

## Notes

- The **spam label must already exist** in the target repository (the bot adds
  it but does not create it). For `google/adk-go` it does.
- This is a moderation aid, not a verdict: it flags and notifies for human
  review rather than deleting or blocking. It deliberately errs toward inaction —
  the prompt instructs the model not to flag merely unhelpful, off-topic, or
  beginner content.

## Known limitations

- **Truncation padding.** Each snippet is truncated to ~1500 runes before review,
  so a determined spammer can pad with benign text ahead of a spam link to push
  it past the cutoff. A production system would prioritize link-bearing regions
  (or raise the cap); this sample keeps the simple bound.
- **Author association is a prior, not proof.** It nudges borderline calls; it
  is not a substitute for reading the content, and spam from an established
  account is still flagged on its merits.
