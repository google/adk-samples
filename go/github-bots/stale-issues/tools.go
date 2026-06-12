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
	"errors"
	"fmt"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

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

// tools builds the function tools the agent uses. The names match those
// referenced by the prompt's decision tree. Each handler closes over the
// GitHub client; agent.ToolContext embeds context.Context, so it is passed
// directly to the API calls.
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
	}, func(ctx agent.ToolContext, a issueArg) (IssueState, error) {
		return c.GetIssueState(ctx, a.IssueNumber)
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "add_label_to_issue",
		Description: "Adds the specified label to the issue.",
	}, func(ctx agent.ToolContext, a labelArg) (actionResult, error) {
		if err := c.AddLabel(ctx, a.IssueNumber, a.Label); err != nil {
			return actionResult{}, err
		}
		return okResult, nil
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "remove_label_from_issue",
		Description: "Removes the specified label from the issue.",
	}, func(ctx agent.ToolContext, a labelArg) (actionResult, error) {
		if err := c.RemoveLabel(ctx, a.IssueNumber, a.Label); err != nil {
			return actionResult{}, err
		}
		return okResult, nil
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "add_stale_label_and_comment",
		Description: "Marks the issue as stale by adding the stale label and posting an explanatory comment.",
	}, func(ctx agent.ToolContext, a issueArg) (actionResult, error) {
		comment := fmt.Sprintf(
			"This issue has been automatically marked as stale because it has not had recent "+
				"activity for %s days after a maintainer requested clarification. It will be "+
				"closed if no further activity occurs within %s days.",
			formatDays(c.cfg.StaleAfter), formatDays(c.cfg.CloseAfter),
		)
		if err := c.MarkStale(ctx, a.IssueNumber, comment); err != nil {
			return actionResult{}, err
		}
		return okResult, nil
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "alert_maintainer_of_edit",
		Description: "Posts a comment alerting maintainers that the author silently edited the issue description.",
	}, func(ctx agent.ToolContext, a issueArg) (actionResult, error) {
		// The body must start with botAlertSignature so the bot recognizes its
		// own alert on future runs and avoids spamming.
		if err := c.Comment(ctx, a.IssueNumber, botAlertSignature+". Maintainers, please review."); err != nil {
			return actionResult{}, err
		}
		return okResult, nil
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "close_as_stale",
		Description: "Closes the issue as not planned after it has remained stale past the close threshold.",
	}, func(ctx agent.ToolContext, a issueArg) (actionResult, error) {
		comment := fmt.Sprintf(
			"This has been automatically closed because it has been marked as stale for over %s days.",
			formatDays(c.cfg.CloseAfter),
		)
		if err := c.CloseAsStale(ctx, a.IssueNumber, comment); err != nil {
			return actionResult{}, err
		}
		return okResult, nil
	}))

	if len(errs) > 0 {
		return nil, fmt.Errorf("create tools: %w", errors.Join(errs...))
	}
	return tools, nil
}
