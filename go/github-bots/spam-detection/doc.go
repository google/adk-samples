// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Command githubspambot is an autonomous ADK Go agent that moderates a GitHub
// repository's issues for spam.
//
// It audits open issues for SEO spam, unsolicited promotion of third-party
// products or sites, and other off-topic solicitation. When the model judges an
// issue's content to be spam it applies a configurable label (default "spam")
// and posts a single comment alerting the maintainers.
//
// The bot is code-orchestrated: code finds the candidate issues and runs the
// LLMAgent once per issue in its own isolated session, with bounded concurrency
// via errgroup. Each issue's session is scoped to that issue (withAuditedIssue),
// so injected instructions in the (untrusted) issue or comment text cannot make
// a tool act on a different issue. The deterministic pre-processing is done
// entirely in code and the finished, untrusted-marked text is embedded in the
// per-issue prompt; the model's only tool is flag_issue_as_spam. Before the
// model is ever invoked, the bot:
//   - skips issues already labeled spam or already carrying the bot's alert
//     comment;
//   - drops comments from maintainers, "[bot]" accounts, and its own identity;
//   - truncates long text (it does not strip fenced code blocks, so spam cannot
//     hide inside a ``` fence).
//
// If nothing reviewable remains, the issue is skipped without spending a single
// model token (the "zero-waste" optimization from the original Python sample).
// Only the fuzzy spam classification is delegated to the model. A -dry-run flag
// logs intended actions without mutating anything.
//
// Idempotency: a re-run never duplicates work. The spam label is the primary
// guard (the sweep excludes already-labeled issues and the per-issue check skips
// them); within a run, flagging an issue twice is a no-op in code; the bot's own
// alert comment is a best-effort secondary signal (see github.go for its bound).
//
// Deliberate differences from the Python adk_issue_monitoring_agent original:
// the scheduled sweep is capped (ISSUE_COUNT, most-recently-updated first) and
// meant as a backstop to the workflow's issue/issue_comment event triggers,
// rather than Python's uncapped 24h "since" sweep / INITIAL_FULL_SCAN; the issue
// title is reviewed in addition to the body; fenced code blocks are kept (Python
// stripped them) so spam cannot hide in a code fence; the alert is plain text
// rather than an @maintainers mention; and bounded concurrency replaces Python's
// inter-batch sleep. See README.md for the full list.
//
// The agent is designed to run on a schedule (and on issue/comment events) from
// a GitHub Actions workflow using the built-in GITHUB_TOKEN. See README.md.
package main
