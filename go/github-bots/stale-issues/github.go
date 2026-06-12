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
	"log/slog"
	"time"

	"github.com/google/go-github/v66/github"
)

// ErrIssueNotFound is returned when the requested issue does not exist.
var ErrIssueNotFound = errors.New("issue not found")

// GitHubClient wraps the go-github REST client and adds a raw GraphQL helper,
// the bot's resolved identity, and dry-run handling. It carries no
// package-level state so it can be constructed per run and unit-tested with an
// httptest server.
type GitHubClient struct {
	rest      *github.Client
	cfg       *Config
	selfLogin string
	log       *slog.Logger
}

// NewGitHubClient builds a client authenticated with the configured token and
// resolves the bot's own login (used to ignore the bot's own activity).
func NewGitHubClient(ctx context.Context, cfg *Config, log *slog.Logger) (*GitHubClient, error) {
	rest := github.NewClient(nil).WithAuthToken(cfg.GitHubToken)
	c := &GitHubClient{rest: rest, cfg: cfg, log: log}

	// Resolve identity once. github-actions[bot] already ends in "[bot]" (so
	// the timeline filter ignores it), but resolving the login makes the bot
	// robust to any token identity.
	if u, _, err := rest.Users.Get(ctx, ""); err == nil {
		c.selfLogin = u.GetLogin()
		log.Info("resolved bot identity", "login", c.selfLogin)
	} else {
		log.Warn("could not resolve bot identity; relying on [bot] suffix filtering", "error", err)
	}
	return c, nil
}

// SearchOldOpenIssues returns the numbers of open issues created before the
// stale threshold, using the Search API to avoid scanning recent issues. PRs
// are excluded.
//
// Because candidates are restricted to issues older than the stale threshold,
// the silent-edit alert path only ever runs on older issues; a description edit
// on a very recent issue is not detected. A transient Search rate-limit error
// surfaces as an error and aborts this run (the next scheduled run retries).
func (c *GitHubClient) SearchOldOpenIssues(ctx context.Context) ([]int, error) {
	cutoff := time.Now().UTC().Add(-c.cfg.StaleAfter).Format("2006-01-02T15:04:05Z")
	query := fmt.Sprintf("repo:%s/%s is:issue state:open created:<%s", c.cfg.Owner, c.cfg.Repo, cutoff)
	c.log.Info("searching for stale candidates", "query", query)

	opts := &github.SearchOptions{
		Sort:        "created",
		Order:       "asc",
		ListOptions: github.ListOptions{PerPage: 100},
	}
	var numbers []int
	for {
		result, resp, err := c.rest.Search.Issues(ctx, query, opts)
		if err != nil {
			return nil, fmt.Errorf("search issues: %w", err)
		}
		for _, issue := range result.Issues {
			if issue.IsPullRequest() {
				continue
			}
			numbers = append(numbers, issue.GetNumber())
		}
		if resp.NextPage == 0 {
			break
		}
		opts.Page = resp.NextPage
	}
	c.log.Info("found stale candidates", "count", len(numbers))
	return numbers, nil
}

// graphQLResponse is the envelope returned by the GitHub GraphQL API.
type graphQLResponse struct {
	Data struct {
		Repository struct {
			Issue *rawIssue `json:"issue"`
		} `json:"repository"`
	} `json:"data"`
	Errors []struct {
		Message string `json:"message"`
	} `json:"errors"`
}

const issueHistoryQuery = `
query($owner: String!, $name: String!, $number: Int!, $commentLimit: Int!, $editLimit: Int!, $timelineLimit: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      author { login }
      createdAt
      labels(first: 20) { nodes { name } }
      comments(last: $commentLimit) {
        nodes { author { login } body createdAt lastEditedAt }
      }
      userContentEdits(last: $editLimit) {
        nodes { editor { login } editedAt }
      }
      timelineItems(itemTypes: [LABELED_EVENT, RENAMED_TITLE_EVENT, REOPENED_EVENT], last: $timelineLimit) {
        nodes {
          __typename
          ... on LabeledEvent { createdAt actor { login } label { name } }
          ... on RenamedTitleEvent { createdAt actor { login } }
          ... on ReopenedEvent { createdAt actor { login } }
        }
      }
    }
  }
}`

// FetchIssueHistory retrieves an issue's full history in a single GraphQL query,
// issued through the authenticated go-github client (no extra dependency). The
// response decodes into rawIssue (defined in state.go).
func (c *GitHubClient) FetchIssueHistory(ctx context.Context, number int) (*rawIssue, error) {
	body := map[string]any{
		"query": issueHistoryQuery,
		"variables": map[string]any{
			"owner":  c.cfg.Owner,
			"name":   c.cfg.Repo,
			"number": number,
			// Bounded windows keep the query cheap. They are generous enough
			// that the stale LabeledEvent and the bot's own alert comment are
			// usually still in view; computeIssueState degrades gracefully when
			// they are not.
			"commentLimit":  50,
			"editLimit":     10,
			"timelineLimit": 50,
		},
	}
	req, err := c.rest.NewRequest("POST", "graphql", body)
	if err != nil {
		return nil, fmt.Errorf("build graphql request: %w", err)
	}
	var out graphQLResponse
	if _, err := c.rest.Do(ctx, req, &out); err != nil {
		return nil, fmt.Errorf("graphql request: %w", err)
	}
	if len(out.Errors) > 0 {
		return nil, fmt.Errorf("graphql error: %s", out.Errors[0].Message)
	}
	if out.Data.Repository.Issue == nil {
		return nil, fmt.Errorf("issue #%d: %w", number, ErrIssueNotFound)
	}
	return out.Data.Repository.Issue, nil
}

// GetIssueState fetches and analyzes an issue, returning the structured summary
// consumed by the agent.
func (c *GitHubClient) GetIssueState(ctx context.Context, number int) (IssueState, error) {
	raw, err := c.FetchIssueHistory(ctx, number)
	if err != nil {
		return IssueState{}, err
	}
	return computeIssueState(raw, c.selfLogin, c.cfg.Maintainers, c.cfg.StaleLabel, c.cfg.StaleAfter, c.cfg.CloseAfter, time.Now().UTC()), nil
}

// --- Mutations (all honor dry-run) ------------------------------------------

// AddLabel adds a label to the issue. It is a no-op under dry-run.
func (c *GitHubClient) AddLabel(ctx context.Context, number int, label string) error {
	if c.shouldSkip(number, "add label %q", label) {
		return nil
	}
	_, _, err := c.rest.Issues.AddLabelsToIssue(ctx, c.cfg.Owner, c.cfg.Repo, number, []string{label})
	return err
}

// RemoveLabel removes a label from the issue. It is a no-op under dry-run.
func (c *GitHubClient) RemoveLabel(ctx context.Context, number int, label string) error {
	if c.shouldSkip(number, "remove label %q", label) {
		return nil
	}
	_, err := c.rest.Issues.RemoveLabelForIssue(ctx, c.cfg.Owner, c.cfg.Repo, number, label)
	return err
}

// Comment posts a comment on the issue. It is a no-op under dry-run.
func (c *GitHubClient) Comment(ctx context.Context, number int, body string) error {
	if c.shouldSkip(number, "comment") {
		return nil
	}
	_, _, err := c.rest.Issues.CreateComment(ctx, c.cfg.Owner, c.cfg.Repo, number, &github.IssueComment{Body: github.String(body)})
	return err
}

// MarkStale adds the stale label, then posts the warning comment. The label is
// applied first so a failure mid-operation cannot leave the issue commented but
// unlabeled (which would cause a duplicate comment on the next run).
func (c *GitHubClient) MarkStale(ctx context.Context, number int, comment string) error {
	if err := c.AddLabel(ctx, number, c.cfg.StaleLabel); err != nil {
		return fmt.Errorf("add stale label: %w", err)
	}
	if err := c.Comment(ctx, number, comment); err != nil {
		return fmt.Errorf("post stale comment: %w", err)
	}
	return nil
}

// CloseAsStale closes the issue as "not planned" (rather than the default
// "completed") and then posts a closing comment.
//
// The issue is closed before the comment is posted: closing is idempotent and,
// once closed, the issue drops out of the next run's open-issue search, so a
// failed comment can never produce a duplicate closing comment on retry.
func (c *GitHubClient) CloseAsStale(ctx context.Context, number int, comment string) error {
	if !c.shouldSkip(number, "close as not_planned") {
		if _, _, err := c.rest.Issues.Edit(ctx, c.cfg.Owner, c.cfg.Repo, number, &github.IssueRequest{
			State:       github.String("closed"),
			StateReason: github.String("not_planned"),
		}); err != nil {
			return fmt.Errorf("close issue: %w", err)
		}
	}
	if err := c.Comment(ctx, number, comment); err != nil {
		return fmt.Errorf("post closing comment: %w", err)
	}
	return nil
}

// shouldSkip logs an intended mutation and reports whether it should be skipped
// because dry-run is enabled.
func (c *GitHubClient) shouldSkip(number int, format string, args ...any) bool {
	if c.cfg.DryRun {
		c.log.Info("[dry-run] would "+fmt.Sprintf(format, args...), "issue", number)
		return true
	}
	return false
}
