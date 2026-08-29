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
	"strings"
	"sync"
	"time"

	"github.com/google/go-github/v66/github"
)

// ErrIssueNotFound is returned when an issue does not exist or refers to a pull
// request. GitHub signals this either with a null repository.issue or a
// GraphQL error of type NOT_FOUND; GetIssue maps both to this sentinel.
var ErrIssueNotFound = errors.New("issue not found")

// errNotApplied reports a write GitHub accepted (HTTP 200) but did not apply,
// which happens when the token lacks push access. It is distinguishable from a
// transport error on purpose: only this case is safe to retry, because after a
// transport error the write may have landed and only its response was lost.
var errNotApplied = errors.New("github did not apply the change")

// maxSearchPages bounds GraphQL search pagination as a safety valve.
// GraphQL page sizes. Named so the query and its variables cannot drift apart,
// and so the cost of widening one is visible at the call site.
const (
	// labelPageSize bounds the labels fetched per issue. Issues carrying more
	// labels than this are rare, and only the categorization labels matter.
	labelPageSize = 20
	// searchPageSize bounds one page of search results.
	searchPageSize = 50
)

const maxSearchPages = 10

// Client wraps the GitHub REST and GraphQL APIs. All mutations route through
// shouldSkip so dry-run is impossible to forget.
type Client struct {
	rest *github.Client
	cfg  *Config
	log  *slog.Logger

	// authorized maps each issue number the agent may mutate to the fields it
	// still needs. It is the defense against prompt injection: a malicious issue
	// body cannot make the agent act on an arbitrary issue, because only issues
	// the bot legitimately targeted (the single -issue, or those returned by
	// list_untriaged_issues) are authorized — and only for the fields that are
	// actually missing, so an already-set type or label can't be overwritten.
	// Guarded by mu because the framework may execute tool calls concurrently.
	mu         sync.Mutex
	authorized map[int]need
	// toolErrored records whether any tool hit an infrastructure (non-validation)
	// error during the run, so the program can exit non-zero and CI fails loudly
	// even though such errors are also handed back to the model as data.
	toolErrored bool
}

// NewClient builds an authenticated GitHub client.
func NewClient(cfg *Config, log *slog.Logger) *Client {
	return &Client{
		rest:       github.NewClient(nil).WithAuthToken(cfg.GitHubToken),
		cfg:        cfg,
		log:        log,
		authorized: make(map[int]need),
	}
}

// authorize records that an issue may be mutated, for the given missing fields.
// It merges with any existing authorization so a repeated list cannot resurrect
// a need already satisfied this run (which would re-enable an overwrite): a field
// stays needed only if it was needed before and still is.
func (c *Client) authorize(number int, n need) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.authorized == nil {
		c.authorized = make(map[int]need)
	}
	if existing, ok := c.authorized[number]; ok {
		n.typ = n.typ && existing.typ
		n.label = n.label && existing.label
	}
	c.authorized[number] = n
}

// claimType atomically reserves an issue's type need for a single mutation. It
// reports whether the issue is authorized at all, and whether this call won the
// reservation (the type was still needed and is now marked satisfied). Reserving
// before the network write is what makes the no-overwrite guarantee hold under
// the framework's concurrent tool execution: of several same-issue calls in one
// turn, exactly one can claim the need and reach the API. If the subsequent
// write fails, the caller must releaseType so the field can be retried.
func (c *Client) claimType(number int) (claimed, authorized bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	n, ok := c.authorized[number]
	if !ok {
		return false, false
	}
	if !n.typ {
		return false, true
	}
	n.typ = false
	c.authorized[number] = n
	return true, true
}

// releaseType restores a type need reserved by claimType, after a failed write.
func (c *Client) releaseType(number int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if n, ok := c.authorized[number]; ok {
		n.typ = true
		c.authorized[number] = n
	}
}

// claimLabel atomically reserves an issue's label need for a single mutation,
// with the same contract as claimType.
func (c *Client) claimLabel(number int) (claimed, authorized bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	n, ok := c.authorized[number]
	if !ok {
		return false, false
	}
	if !n.label {
		return false, true
	}
	n.label = false
	c.authorized[number] = n
	return true, true
}

// releaseLabel restores a label need reserved by claimLabel, after a failed add.
func (c *Client) releaseLabel(number int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if n, ok := c.authorized[number]; ok {
		n.label = true
		c.authorized[number] = n
	}
}

// recordToolError flags that a tool hit an infrastructure error this run.
func (c *Client) recordToolError() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.toolErrored = true
}

// hadToolError reports whether any tool hit an infrastructure error this run.
func (c *Client) hadToolError() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.toolErrored
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

var issueFields = fmt.Sprintf(`
		number
		title
		body
		issueType { name }
		labels(first: %d) { nodes { name } }`, labelPageSize)

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
	// Dedupe across pages, as both siblings do. An issue whose position shifts
	// between cursor fetches can appear on two pages, and each copy would get its
	// own agent session -- a wasted model call, and one fewer distinct issue
	// triaged than `count` promises.
	seen := make(map[int]bool)
	for page := 0; page < maxSearchPages && len(out) < count; page++ {
		vars := map[string]any{"q": q, "first": searchPageSize}
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
			if seen[iss.Number] {
				continue
			}
			if needsTriage(iss, c.cfg.AllowedLabels).any() {
				seen[iss.Number] = true
				out = append(out, iss)
				if len(out) >= count {
					break
				}
			}
		}
		// Stop if there's no next page, or if the cursor is missing (guards
		// against re-requesting the same page on a malformed response).
		if !resp.Data.Search.PageInfo.HasNextPage || resp.Data.Search.PageInfo.EndCursor == "" {
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
		return fmt.Errorf("build set-type request (nothing was sent): %w: %w", errNotApplied, err)
	}
	// Setting a type requires push access; without it GitHub returns 200 with the
	// issue unchanged, silently dropping the type. Read back the response and
	// confirm the type was actually applied so a no-op can't masquerade as success.
	// (The REST payload names the field "type", unlike GraphQL's "issueType".)
	var updated struct {
		Type *struct {
			Name string `json:"name"`
		} `json:"type"`
	}
	if _, err := c.rest.Do(ctx, req, &updated); err != nil {
		return fmt.Errorf("set issue type: %w", err)
	}
	if updated.Type == nil || !strings.EqualFold(updated.Type.Name, issueType) {
		return fmt.Errorf("set issue type %q on issue #%d (the token likely lacks push access): %w", issueType, number, errNotApplied)
	}
	c.log.Info("set issue type", "issue", number, "type", issueType)
	return nil
}

// AddLabel adds a single label to the issue.
func (c *Client) AddLabel(ctx context.Context, number int, label string) error {
	if c.shouldSkip(number, "add label %q", label) {
		return nil
	}
	// AddLabelsToIssue returns the issue's labels after the add. Like types,
	// labels are silently dropped without push access, so confirm ours is present
	// rather than trusting a 200.
	labels, _, err := c.rest.Issues.AddLabelsToIssue(ctx, c.cfg.Owner, c.cfg.Repo, number, []string{label})
	if err != nil {
		return fmt.Errorf("add label: %w", err)
	}
	applied := false
	for _, l := range labels {
		if strings.EqualFold(l.GetName(), label) {
			applied = true
			break
		}
	}
	if !applied {
		return fmt.Errorf("add label %q to issue #%d (the token likely lacks push access): %w", label, number, errNotApplied)
	}
	c.log.Info("added label", "issue", number, "label", label)
	return nil
}
