# ADK Go GitHub Bots

GitHub-automation bots built with the [Agent Development Kit (ADK) for
Go](https://github.com/google/adk-go). Each bot is a self-contained Go module
that an agent uses to take real actions on a GitHub repository (labeling,
typing, commenting, closing issues, …) by reasoning over issue content and
calling typed tools.

These samples double as **runnable automation**: each ships a *reusable
workflow* so another repository can run the bot on **its own** issues without
copying the code, simply by adding a small caller workflow that references it.

## Bots

| Bot | Description |
| --- | --- |
| [`issue-triage`](./issue-triage) | Sets each open issue's type (Bug/Feature/Task) and a categorization label. |
| [`stale-issues`](./stale-issues) | Audits open issues for staleness; warns then closes issues left waiting on the author. |

## How the cross-repo wiring works

Each bot provides a reusable workflow in this repo
(`.github/workflows/go-<bot>.yml`, `on: workflow_call`). A *target* repository
adds a thin **caller** workflow with its own triggers (`issues`, `schedule`,
`workflow_dispatch`) that does:

```yaml
jobs:
  run:
    permissions: { issues: write, contents: read }
    uses: google/adk-samples/.github/workflows/go-<bot>.yml@<commit-sha>
    with: { ... }
    secrets: { GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }} }
```

A reusable workflow runs in the **caller's** context, so the caller's built-in
`GITHUB_TOKEN` (scoped by its `permissions:` block) writes to the *target* repo —
no PAT or GitHub App required. See each bot's README for details and the exact
caller snippet.

## Running locally

Each bot is an independent module — `cd` into it, copy `.env.example` to `.env`,
and follow its README.
