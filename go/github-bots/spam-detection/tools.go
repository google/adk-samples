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

package main

import (
	"context"
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// auditedIssueKey scopes a session to a single issue number. The runner builds
// the invocation context from the context passed to Run (which embeds it), so a
// value set here is visible to every tool via ctx.Value.
type auditedIssueKey struct{}

// withAuditedIssue binds the issue this session is allowed to mutate.
func withAuditedIssue(ctx context.Context, number int) context.Context {
	return context.WithValue(ctx, auditedIssueKey{}, number)
}

// authorizeIssue reports whether the tool may act on the requested issue. It is
// the defense against prompt injection: untrusted issue or comment text cannot
// make the agent flag an issue other than the one this session is reviewing.
//
// This relies on ADK propagating the context passed to runner.Run (which carries
// the withAuditedIssue value) through to the agent.Context seen here. If a
// future ADK release stops embedding context in its tool context, the lookup
// below simply misses and every call is rejected — the bot stops flagging
// (fail-safe) rather than acting on the wrong issue.
func authorizeIssue(ctx context.Context, requested int) (string, bool) {
	audited, ok := ctx.Value(auditedIssueKey{}).(int)
	if !ok {
		return "no issue is authorized for this session", false
	}
	if requested != audited {
		return fmt.Sprintf("session is scoped to issue #%d; refusing to act on issue #%d", audited, requested), false
	}
	return "", true
}

// flagArgs is the input for the flag_issue_as_spam tool. functiontool.New
// reflects over this struct to build the JSON schema the model fills, so the
// json tags name the fields the model produces.
type flagArgs struct {
	IssueNumber     int    `json:"issue_number"`
	DetectionReason string `json:"detection_reason"`
}

// actionResult is the typed result returned by the tool.
type actionResult struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

var okResult = actionResult{Status: "success", Message: "maintainers alerted"}

// errResult is a model-readable failure: the tool ran as a Go call but the
// request was rejected (e.g. wrong issue). It is returned with a nil Go error so
// the model receives it as data. Real I/O failures return a Go error instead.
func errResult(format string, a ...any) actionResult {
	return actionResult{Status: "error", Message: fmt.Sprintf(format, a...)}
}

// flagAsSpam is the body of the flag_issue_as_spam tool, factored out so it can
// be unit-tested without going through the agent. An issue mismatch is returned
// as a model-readable errResult (nil Go error); an I/O failure returns a Go
// error and is recorded so the run fails loudly.
func (c *GitHubClient) flagAsSpam(ctx context.Context, number int, reason string) (actionResult, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	// Enforce "flag at most once per issue" in code, not just in the prompt: if
	// the model emits the tool twice, the second call is a no-op so it cannot
	// post a duplicate alert comment.
	if !c.markFlagged(number) {
		return actionResult{Status: "success", Message: "issue already flagged this run"}, nil
	}
	if err := c.FlagSpam(ctx, number, buildAlertComment(reason)); err != nil {
		c.recordError()
		return actionResult{}, err
	}
	return okResult, nil
}

// tools builds the agent's toolset. The bot exposes a single action: flag an
// issue as spam. Everything else (which issues to look at, what text to review)
// is decided in code before the model runs.
func (c *GitHubClient) tools() ([]tool.Tool, error) {
	t, err := functiontool.New(functiontool.Config{
		Name: "flag_issue_as_spam",
		Description: "Flags the issue as spam: applies the spam label and posts a comment " +
			"alerting the maintainers. Call this only when the reviewed content is clearly " +
			"spam. Provide a brief detection_reason explaining what is spam and why.",
	}, func(ctx agent.Context, a flagArgs) (actionResult, error) {
		return c.flagAsSpam(ctx, a.IssueNumber, a.DetectionReason)
	})
	if err != nil {
		return nil, fmt.Errorf("create tools: %w", err)
	}
	return []tool.Tool{t}, nil
}
