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
	"errors"
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
// the defense against prompt injection: untrusted issue content cannot make the
// agent mutate an issue other than the one this session is auditing.
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

// isManagedLabel reports whether the bot is allowed to add/remove this label.
// It only ever manages the stale and request-clarification labels.
func (c *GitHubClient) isManagedLabel(label string) bool {
	return label == c.cfg.StaleLabel || label == c.cfg.RequestClarificationLabel
}

func errResult(format string, a ...any) actionResult {
	return actionResult{Status: "error", Message: fmt.Sprintf(format, a...)}
}

// issueArg is the input for tools that operate on a single issue.
type issueArg struct {
	IssueNumber int `json:"issue_number"`
}

// labelArg is the input for label tools.
type labelArg struct {
	IssueNumber int    `json:"issue_number"`
	Label       string `json:"label"`
}

// actionResult is the typed result returned by mutating tools.
type actionResult struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

var okResult = actionResult{Status: "success"}

// The do* methods below are the tool handler bodies, extracted so they are
// directly unit-testable (the functiontool closures are thin wrappers). Each
// enforces per-issue authorization first; validation failures return a
// model-readable errResult with a nil Go error, while infrastructure failures
// record a tool error (so the run fails loudly) and return the Go error.

// doGetIssueState fetches an issue's state. It is authorized like the mutating
// tools so untrusted content cannot make the model pull an out-of-scope issue's
// data into context.
func (c *GitHubClient) doGetIssueState(ctx context.Context, number int) (IssueState, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return IssueState{Status: "error", LastActionType: msg}, nil
	}
	st, err := c.GetIssueState(ctx, number)
	if err != nil {
		c.recordToolError()
		return IssueState{}, err
	}
	return st, nil
}

func (c *GitHubClient) doAddLabel(ctx context.Context, number int, label string) (actionResult, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	if !c.isManagedLabel(label) {
		return errResult("label %q is not managed by this bot", label), nil
	}
	if err := c.AddLabel(ctx, number, label); err != nil {
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult, nil
}

func (c *GitHubClient) doRemoveLabel(ctx context.Context, number int, label string) (actionResult, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	if !c.isManagedLabel(label) {
		return errResult("label %q is not managed by this bot", label), nil
	}
	if err := c.RemoveLabel(ctx, number, label); err != nil {
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult, nil
}

func (c *GitHubClient) doMarkStale(ctx context.Context, number int) (actionResult, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	comment := fmt.Sprintf(
		"This issue has been automatically marked as stale because it has not had recent "+
			"activity for %s days after a maintainer requested clarification. It will be "+
			"closed if no further activity occurs within %s days.",
		formatDays(c.cfg.StaleAfter), formatDays(c.cfg.CloseAfter),
	)
	if err := c.MarkStale(ctx, number, comment); err != nil {
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult, nil
}

func (c *GitHubClient) doAlertEdit(ctx context.Context, number int) (actionResult, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	// The body must start with botAlertSignature so the bot recognizes its own
	// alert on future runs and avoids spamming.
	if err := c.Comment(ctx, number, botAlertSignature+". Maintainers, please review."); err != nil {
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult, nil
}

func (c *GitHubClient) doClose(ctx context.Context, number int) (actionResult, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	comment := fmt.Sprintf(
		"This has been automatically closed because it has been marked as stale for over %s days.",
		formatDays(c.cfg.CloseAfter),
	)
	if err := c.CloseAsStale(ctx, number, comment); err != nil {
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult, nil
}

// tools builds the function tools the agent uses. The names match those
// referenced by the prompt's decision tree. Each handler closes over the
// GitHub client; agent.Context embeds context.Context, so it is passed
// directly to the do* methods.
func (c *GitHubClient) tools() ([]tool.Tool, error) {
	var (
		tools []tool.Tool
		errs  []error
	)
	add := func(t tool.Tool, err error) {
		if err != nil {
			errs = append(errs, err)
			return
		}
		tools = append(tools, t)
	}

	add(functiontool.New(functiontool.Config{
		Name:        "get_issue_state",
		Description: "Fetches and analyzes the full state of a GitHub issue, returning its staleness, last actor role, labels, and timing.",
	}, func(ctx agent.Context, a issueArg) (IssueState, error) {
		return c.doGetIssueState(ctx, a.IssueNumber)
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "add_label_to_issue",
		Description: "Adds the specified label to the issue.",
	}, func(ctx agent.Context, a labelArg) (actionResult, error) {
		return c.doAddLabel(ctx, a.IssueNumber, a.Label)
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "remove_label_from_issue",
		Description: "Removes the specified label from the issue.",
	}, func(ctx agent.Context, a labelArg) (actionResult, error) {
		return c.doRemoveLabel(ctx, a.IssueNumber, a.Label)
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "add_stale_label_and_comment",
		Description: "Marks the issue as stale by adding the stale label and posting an explanatory comment.",
	}, func(ctx agent.Context, a issueArg) (actionResult, error) {
		return c.doMarkStale(ctx, a.IssueNumber)
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "alert_maintainer_of_edit",
		Description: "Posts a comment alerting maintainers that the author silently edited the issue description.",
	}, func(ctx agent.Context, a issueArg) (actionResult, error) {
		return c.doAlertEdit(ctx, a.IssueNumber)
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "close_as_stale",
		Description: "Closes the issue as not planned after it has remained stale past the close threshold.",
	}, func(ctx agent.Context, a issueArg) (actionResult, error) {
		return c.doClose(ctx, a.IssueNumber)
	}))

	if len(errs) > 0 {
		return nil, fmt.Errorf("create tools: %w", errors.Join(errs...))
	}
	return tools, nil
}
