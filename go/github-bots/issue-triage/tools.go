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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// Tool argument and result types. functiontool.New reflects over these structs
// to build the JSON schema the model sees and fills — the parameter struct IS
// the tool's input contract. The json tags name the fields the LLM produces.

type listArgs struct {
	Count int `json:"count"`
}

type listResult struct {
	Status string  `json:"status"`
	Issues []Issue `json:"issues"`
}

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

// errResult is a *model-readable* failure: the tool call succeeded as a Go
// call, but the requested action was rejected (e.g. a disallowed label). It is
// returned with a nil Go error so the model receives it as data and can correct
// itself. Reserve real Go errors for infrastructure failures (network, API).
func errResult(format string, a ...any) actionResult {
	return actionResult{Status: "error", Message: fmt.Sprintf(format, a...)}
}

// doList fetches untriaged issues and authorizes them for subsequent mutation.
func (c *Client) doList(ctx context.Context, count int) (listResult, error) {
	if count <= 0 || count > c.cfg.IssueCount {
		count = c.cfg.IssueCount
	}
	issues, err := c.ListUntriaged(ctx, count)
	if err != nil {
		return listResult{}, err
	}
	for _, iss := range issues {
		c.authorize(iss.Number)
	}
	return listResult{Status: "success", Issues: issues}, nil
}

// doChangeType validates and applies an issue-type change. Validation and
// authorization failures are returned as model-readable errResults (nil Go
// error); only I/O failures return a Go error.
func (c *Client) doChangeType(ctx context.Context, number int, issueType string) (actionResult, error) {
	if number <= 0 {
		return errResult("invalid issue number %d", number), nil
	}
	if !isValidType(issueType) {
		return errResult("issue type %q is not allowed; use one of: %s", issueType, strings.Join(allowedTypes, ", ")), nil
	}
	if !c.isAuthorized(number) {
		return errResult("issue #%d is not part of the current triage set; only triage issues you fetched", number), nil
	}
	if err := c.SetType(ctx, number, issueType); err != nil {
		return actionResult{}, err
	}
	return okResult("set issue #%d type to %s", number, issueType), nil
}

// doAddLabel validates and applies a label addition, with the same error
// conventions as doChangeType.
func (c *Client) doAddLabel(ctx context.Context, number int, label string) (actionResult, error) {
	if number <= 0 {
		return errResult("invalid issue number %d", number), nil
	}
	if !isAllowedLabel(label, c.cfg.AllowedLabels) {
		return errResult("label %q is not in the allowlist; will not apply", label), nil
	}
	if !c.isAuthorized(number) {
		return errResult("issue #%d is not part of the current triage set; only triage issues you fetched", number), nil
	}
	if err := c.AddLabel(ctx, number, label); err != nil {
		return actionResult{}, err
	}
	return okResult("added label %q to issue #%d", label, number), nil
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
		Name: "list_untriaged_issues",
		Description: "Lists open issues that still need an issue type and/or a categorization label. " +
			"Returns each issue's number, title, body, current labels, and current type.",
	}, func(ctx agent.ToolContext, a listArgs) (listResult, error) {
		return c.doList(ctx, a.Count)
	}))

	add(functiontool.New(functiontool.Config{
		Name: "change_issue_type",
		Description: "Sets the GitHub issue type for an issue. Allowed values: " +
			strings.Join(allowedTypes, ", ") + ".",
	}, func(ctx agent.ToolContext, a typeArgs) (actionResult, error) {
		return c.doChangeType(ctx, a.IssueNumber, a.IssueType)
	}))

	add(functiontool.New(functiontool.Config{
		Name: "add_label_to_issue",
		Description: "Adds one categorization label to an issue. Allowed labels: " +
			strings.Join(c.cfg.AllowedLabels, ", ") + ".",
	}, func(ctx agent.ToolContext, a labelArgs) (actionResult, error) {
		return c.doAddLabel(ctx, a.IssueNumber, a.Label)
	}))

	if len(errs) > 0 {
		return nil, fmt.Errorf("create tools: %w", errors.Join(errs...))
	}
	return tools, nil
}
