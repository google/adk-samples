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
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// Tool argument and result types. functiontool.New reflects over these structs
// to build the JSON schema the model sees and fills — the parameter struct IS
// the tool's input contract. The json tags name the fields the LLM produces.

type typeArgs struct {
	IssueNumber int    `json:"issue_number"`
	IssueType   string `json:"issue_type"`
}

type labelArgs struct {
	IssueNumber int    `json:"issue_number"`
	Label       string `json:"label"`
}

type actionResult struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

func okResult(format string, a ...any) actionResult {
	return actionResult{Status: "success", Message: fmt.Sprintf(format, a...)}
}

// checkIssueArg runs the two checks every mutating tool shares: the session must
// be scoped to this issue, and the number must be a plausible issue number.
func checkIssueArg(ctx context.Context, number int) (string, bool) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return msg, false
	}
	if number <= 0 {
		return fmt.Sprintf("invalid issue number %d", number), false
	}
	return "", true
}

// errResult is a *model-readable* failure: the tool call succeeded as a Go
// call, but the requested action was rejected (e.g. a disallowed label). It is
// returned with a nil Go error so the model receives it as data and can correct
// itself. Reserve real Go errors for infrastructure failures (network, API).
func errResult(format string, a ...any) actionResult {
	return actionResult{Status: "error", Message: fmt.Sprintf(format, a...)}
}

// auditedIssueKey scopes a session to a single issue number. The runner builds
// the invocation context from the context passed to Run (which embeds it), so a
// value set here is visible to every tool via ctx.Value.
type auditedIssueKey struct{}

// withAuditedIssue binds the issue this session is allowed to mutate.
func withAuditedIssue(ctx context.Context, number int) context.Context {
	return context.WithValue(ctx, auditedIssueKey{}, number)
}

// authorizeIssue reports whether the tool may act on the requested issue.
//
// The bot runs one agent session per issue, so untrusted text in issue A cannot
// reach a session scoped to issue B. This is the same per-session scope the
// stale-issues and spam-detection siblings use; it sits in front of the
// need-claim gate below, which separately stops an already-set field being
// overwritten.
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

// doChangeType validates and applies an issue-type change. Validation and
// authorization failures are returned as model-readable errResults (nil Go
// error); only I/O failures return a Go error.
func (c *Client) doChangeType(ctx context.Context, number int, issueType string) (actionResult, error) {
	if msg, ok := checkIssueArg(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	canonical, ok := canonicalType(issueType)
	if !ok {
		return errResult("issue type %q is not allowed; use one of: %s", issueType, strings.Join(allowedTypes, ", ")), nil
	}
	// Reserve the "type" need atomically before the write, so a concurrent
	// duplicate call for the same issue cannot also pass this gate and overwrite.
	claimed, authorized := c.claimType(number)
	if !authorized {
		return errResult("issue #%d is not part of the current triage set; only triage issues you fetched", number), nil
	}
	if !claimed {
		return errResult("issue #%d already has a type; not overwriting", number), nil
	}
	if err := c.SetType(ctx, number, canonical); err != nil {
		// Only retry a write GitHub demonstrably did not apply. After a transport
		// error the POST may have landed, and releasing would let the model claim
		// the need again with a different value.
		if errors.Is(err, errNotApplied) {
			c.releaseType(number)
		}
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult("set issue #%d type to %s", number, canonical), nil
}

// doAddLabel validates and applies a label addition, with the same error
// conventions as doChangeType.
func (c *Client) doAddLabel(ctx context.Context, number int, label string) (actionResult, error) {
	if msg, ok := checkIssueArg(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	canonical, ok := canonicalLabel(label, c.cfg.AllowedLabels)
	if !ok {
		return errResult("label %q is not in the allowlist; will not apply", label), nil
	}
	// Reserve the "label" need atomically before the write, so a concurrent
	// duplicate call for the same issue cannot also pass this gate and add a
	// second label.
	claimed, authorized := c.claimLabel(number)
	if !authorized {
		return errResult("issue #%d is not part of the current triage set; only triage issues you fetched", number), nil
	}
	if !claimed {
		return errResult("issue #%d already has a categorization label; not adding another", number), nil
	}
	if err := c.AddLabel(ctx, number, canonical); err != nil {
		// Only retry a write GitHub demonstrably did not apply; see doChangeType.
		if errors.Is(err, errNotApplied) {
			c.releaseLabel(number)
		}
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult("added label %q to issue #%d", canonical, number), nil
}

// tools builds the agent's toolset. Construction errors are accumulated and
// joined so a single bad tool reports clearly.
func (c *Client) tools() ([]tool.Tool, error) {
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
		Name: "change_issue_type",
		Description: "Sets the GitHub issue type for an issue. Allowed values: " +
			strings.Join(allowedTypes, ", ") + ".",
	}, func(ctx agent.Context, a typeArgs) (actionResult, error) {
		return c.doChangeType(ctx, a.IssueNumber, a.IssueType)
	}))

	add(functiontool.New(functiontool.Config{
		Name: "add_label_to_issue",
		Description: "Adds one categorization label to an issue. Allowed labels: " +
			strings.Join(c.cfg.AllowedLabels, ", ") + ".",
	}, func(ctx agent.Context, a labelArgs) (actionResult, error) {
		return c.doAddLabel(ctx, a.IssueNumber, a.Label)
	}))

	if len(errs) > 0 {
		return nil, fmt.Errorf("create tools: %w", errors.Join(errs...))
	}
	return tools, nil
}
