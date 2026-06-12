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
	"sync"
	"time"

	"github.com/google/go-github/v66/github"
)

// ErrIssueNotFound is returned when an issue does not exist or refers to a pull
// request (GraphQL repository.issue returns null for PRs).
var ErrIssueNotFound = errors.New("issue not found")

// maxSearchPages bounds GraphQL search pagination as a safety valve.
const maxSearchPages = 10

// Client wraps the GitHub REST and GraphQL APIs. All mutations route through
// shouldSkip so dry-run is impossible to forget.
type Client struct {
	rest *github.Client
	cfg  *Config
	log  *slog.Logger

	// authorized bounds which issue numbers the agent may mutate. It is the
	// defense against prompt injection: a malicious issue body cannot make the
	// agent act on an arbitrary issue, because only issues the bot legitimately
	// targeted (the single -issue, or those returned by list_untriaged_issues)
	// are ever authorized. Guarded by mu because the framework may execute tool
	// calls concurrently.
	mu         sync.Mutex
	authorized map[int]bool
}

// NewClient builds an authenticated GitHub client.
func NewClient(cfg *Config, log *slog.Logger) *Client {
	return &Client{
		rest:       github.NewClient(nil).WithAuthToken(cfg.GitHubToken),
		cfg:        cfg,
		log:        log,
		authorized: make(map[int]bool),
	}
}

// authorize marks issue numbers as eligible for mutation.
func (c *Client) authorize(numbers ...int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.authorized == nil {
		c.authorized = make(map[int]bool)
	}
	for _, n := range numbers {
		c.authorized[n] = true
	}
}

// isAuthorized reports whether an issue number may be mutated.
func (c *Client) isAuthorized(number int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.authorized[number]
}

// shouldSkip is the single dry-run chokepoint for every mutation. It logs the
// intended action and returns true when nothing should be written.
func (c *Client) shouldSkip(number int, format string, args ...any) bool {
	if c.cfg.DryRun {
		c.log.Info("[dry-run] would "+fmt.Sprintf(format, args...), "issue", number)
		return true
	}
	return false
}

// --- GraphQL plumbing ---

const issueFields = `
		number
		title
		body
		issueType { name }
		labels(first: 20) { nodes { name } }`

var issueSearchQuery = `query($q: String!, $first: Int!, $after: String) {
	search(query: $q, type: ISSUE, first: $first, after: $after) {
		pageInfo { hasNextPage endCursor }
		nodes {
			... on Issue {` + issueFields + `
			}
		}
	}
}`

var issueByNumberQuery = `query($owner: String!, $name: String!, $number: Int!) {
	repository(owner: $owner, name: $name) {
		issue(number: $number) {` + issueFields + `
		}
	}
}`

type graphQLError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type issueNode struct {
	Number    int    `json:"number"`
	Title     string `json:"title"`
	Body      string `json:"body"`
	IssueType *struct {
		Name string `json:"name"`
	} `json:"issueType"`
	Labels struct {
		Nodes []struct {
			Name string `json:"name"`
		} `json:"nodes"`
	} `json:"labels"`
}

func (n issueNode) toIssue() Issue {
	labels := make([]string, 0, len(n.Labels.Nodes))
	for _, l := range n.Labels.Nodes {
		labels = append(labels, l.Name)
	}
	var typeName string
	if n.IssueType != nil {
		typeName = n.IssueType.Name
	}
	return Issue{
		Number: n.Number,
		Title:  n.Title,
		// Truncate here so long bodies never bloat the prompt, whether the
		// issue arrives via the batch sweep or a single-issue fetch.
		Body:   truncate(n.Body, maxBodyRunes),
		Labels: labels,
		Type:   typeName,
	}
}

type searchResponse struct {
	Data struct {
		Search struct {
			PageInfo struct {
				HasNextPage bool   `json:"hasNextPage"`
				EndCursor   string `json:"endCursor"`
			} `json:"pageInfo"`
			Nodes []issueNode `json:"nodes"`
		} `json:"search"`
	} `json:"data"`
	Errors []graphQLError `json:"errors"`
}

type issueResponse struct {
	Data struct {
		Repository struct {
			Issue *issueNode `json:"issue"`
		} `json:"repository"`
	} `json:"data"`
	Errors []graphQLError `json:"errors"`
}

// graphQL issues a GraphQL request through the authenticated REST client (a raw
// POST to the /graphql endpoint), decoding the JSON body into out.
func (c *Client) graphQL(ctx context.Context, query string, vars map[string]any, out any) error {
	body := map[string]any{"query": query, "variables": vars}
	req, err := c.rest.NewRequest("POST", "graphql", body)
	if err != nil {
		return fmt.Errorf("build graphql request: %w", err)
	}
	if _, err := c.rest.Do(ctx, req, out); err != nil {
		return fmt.Errorf("graphql request: %w", err)
	}
	return nil
}

// --- Reads ---

// ListUntriaged returns up to count open issues (newest first) that need an
// issue type and/or a categorization label, optionally restricted to a
// freshness window. Pull requests are excluded by querying type:ISSUE.
func (c *Client) ListUntriaged(ctx context.Context, count int) ([]Issue, error) {
	q := fmt.Sprintf("repo:%s/%s is:issue is:open sort:created-desc", c.cfg.Owner, c.cfg.Repo)
	if c.cfg.FreshnessWindow > 0 {
		cutoff := time.Now().UTC().Add(-c.cfg.FreshnessWindow).Format("2006-01-02")
		q += " created:>=" + cutoff
	}

	var (
		out   []Issue
		after string
	)
	for page := 0; page < maxSearchPages && len(out) < count; page++ {
		vars := map[string]any{"q": q, "first": 50}
		if after != "" {
			vars["after"] = after
		}
		var resp searchResponse
		if err := c.graphQL(ctx, issueSearchQuery, vars, &resp); err != nil {
			return nil, err
		}
		if len(resp.Errors) > 0 {
			return nil, fmt.Errorf("graphql search: %s", resp.Errors[0].Message)
		}
		for _, node := range resp.Data.Search.Nodes {
			iss := node.toIssue()
			if iss.Number == 0 {
				continue // not an Issue node
			}
			if needsType, needsLabel := needsTriage(iss, c.cfg.AllowedLabels); needsType || needsLabel {
				out = append(out, iss)
				if len(out) >= count {
					break
				}
			}
		}
		if !resp.Data.Search.PageInfo.HasNextPage {
			break
		}
		after = resp.Data.Search.PageInfo.EndCursor
	}
	return out, nil
}

// GetIssue fetches a single issue by number. It returns ErrIssueNotFound if the
// issue does not exist or is a pull request.
func (c *Client) GetIssue(ctx context.Context, number int) (Issue, error) {
	vars := map[string]any{"owner": c.cfg.Owner, "name": c.cfg.Repo, "number": number}
	var resp issueResponse
	if err := c.graphQL(ctx, issueByNumberQuery, vars, &resp); err != nil {
		return Issue{}, err
	}
	if len(resp.Errors) > 0 {
		// GitHub returns a NOT_FOUND error (not a null issue) when the number
		// does not exist or refers to a pull request.
		for _, e := range resp.Errors {
			if e.Type == "NOT_FOUND" {
				return Issue{}, fmt.Errorf("issue #%d: %w", number, ErrIssueNotFound)
			}
		}
		return Issue{}, fmt.Errorf("graphql issue: %s", resp.Errors[0].Message)
	}
	if resp.Data.Repository.Issue == nil {
		return Issue{}, fmt.Errorf("issue #%d: %w", number, ErrIssueNotFound)
	}
	return resp.Data.Repository.Issue.toIssue(), nil
}

// --- Mutations ---

// SetType sets the GitHub issue type via a raw PATCH (go-github v66 has no typed
// support for issue types yet).
func (c *Client) SetType(ctx context.Context, number int, issueType string) error {
	if c.shouldSkip(number, "set issue type to %q", issueType) {
		return nil
	}
	u := fmt.Sprintf("repos/%s/%s/issues/%d", c.cfg.Owner, c.cfg.Repo, number)
	req, err := c.rest.NewRequest("PATCH", u, map[string]any{"type": issueType})
	if err != nil {
		return fmt.Errorf("build set-type request: %w", err)
	}
	if _, err := c.rest.Do(ctx, req, nil); err != nil {
		return fmt.Errorf("set issue type: %w", err)
	}
	c.log.Info("set issue type", "issue", number, "type", issueType)
	return nil
}

// AddLabel adds a single label to the issue.
func (c *Client) AddLabel(ctx context.Context, number int, label string) error {
	if c.shouldSkip(number, "add label %q", label) {
		return nil
	}
	if _, _, err := c.rest.Issues.AddLabelsToIssue(ctx, c.cfg.Owner, c.cfg.Repo, number, []string{label}); err != nil {
		return fmt.Errorf("add label: %w", err)
	}
	c.log.Info("added label", "issue", number, "label", label)
	return nil
}
