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

const (
	// resolveIdentityTimeout bounds the one-off "who am I" lookup at startup, so
	// a hung request cannot stall the run until the workflow's own timeout.
	resolveIdentityTimeout = 10 * time.Second
	// searchPageSize bounds one page of REST search results. The GraphQL query
	// below carries its own literal limits; they are not derived from this.
	searchPageSize = 100
)

// ErrIssueNotFound is returned when the requested issue does not exist or refers
// to a pull request.
var ErrIssueNotFound = errors.New("issue not found")

// GitHubClient wraps the go-github REST client and adds a raw GraphQL helper,
// the bot's resolved identity, dry-run handling, and run-level error tracking.
// It carries no package-level state so it can be constructed per run and
// unit-tested with an httptest server.
type GitHubClient struct {
	rest      *github.Client
	cfg       *Config
	selfLogin string
	log       *slog.Logger

	// maintainers is the lowercased set of trusted logins, built once from
	// cfg.Maintainers so the concurrent per-issue workers don't each rebuild it.
	maintainers map[string]bool

	// mu guards errored and flagged, which may be touched from tool handlers
	// running concurrently across issues.
	mu sync.Mutex
	// errored records whether any infrastructure error (a failed GitHub
	// mutation, fetch, or agent run) occurred this run, so the program can exit
	// non-zero and a scheduled/CI run fails loudly.
	errored bool
	// flagged records the issues already flagged this run, so a model that emits
	// the flag tool twice for the same issue cannot post a duplicate comment
	// (the label add is idempotent server-side, but a second comment is not).
	flagged map[int]flagOutcome
}

// NewGitHubClient builds a client authenticated with the configured token and
// resolves the bot's own login (used to ignore the bot's own activity and to
// authenticate its prior alert comments).
func NewGitHubClient(ctx context.Context, cfg *Config, log *slog.Logger) (*GitHubClient, error) {
	rest := github.NewClient(nil).WithAuthToken(cfg.GitHubToken)
	c := &GitHubClient{
		rest:        rest,
		cfg:         cfg,
		log:         log,
		flagged:     make(map[int]flagOutcome),
		maintainers: maintainerSet(cfg.Maintainers),
	}

	// Resolve identity once, under a short timeout so a hanging API call can't
	// stall startup indefinitely. Resolving the login makes the bot robust to any
	// token identity (e.g. a PAT); on failure it falls back to suffix filtering.
	idCtx, cancel := context.WithTimeout(ctx, resolveIdentityTimeout)
	defer cancel()
	if u, _, err := rest.Users.Get(idCtx, ""); err == nil {
		c.selfLogin = u.GetLogin()
		log.Info("resolved bot identity", "login", c.selfLogin)
	} else {
		// Fall back to "[bot]" suffix filtering. This is fine for the default
		// github-actions[bot] token; with a non-[bot] PAT the bot can no longer
		// recognize its own past alert comments, so cross-run dedup then rests
		// solely on the spam label (still the primary guard).
		log.Warn("could not resolve bot identity; relying on [bot] suffix filtering", "error", err)
	}
	return c, nil
}

// recordError flags that an infrastructure error occurred this run.
func (c *GitHubClient) recordError() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.errored = true
}

// hadError reports whether any infrastructure error occurred this run.
func (c *GitHubClient) hadError() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.errored
}

// flagOutcome is what happened to an issue's flag attempt this run.
type flagOutcome int

const (
	flagUnattempted flagOutcome = iota
	flagInFlight                // claimed, write not yet known to have succeeded
	flagFailed                  // the write returned an error
)

// markFlagged claims the single flag attempt allowed for an issue this run, and
// reports whether this call won the claim.
//
// The claim is deliberately NOT rolled back on failure: retrying inside the same
// run risks posting the alert twice if the first write actually landed and only
// its response was lost. A failed flag is recorded as an error (fail-loud) and
// retried on the next scheduled run, by which point the label guard applies.
//
// What IS tracked is whether the attempt failed, so a second tool call after a
// failure reports an error rather than "already flagged" — otherwise the model's
// transcript records a success for a comment that was never posted.
func (c *GitHubClient) markFlagged(number int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.flagged == nil {
		c.flagged = make(map[int]flagOutcome)
	}
	if c.flagged[number] != flagUnattempted {
		return false
	}
	c.flagged[number] = flagInFlight
	return true
}

// recordFlagFailure notes that the claimed flag write failed, so a later call
// for the same issue is told the truth instead of "already flagged".
func (c *GitHubClient) recordFlagFailure(number int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.flagged == nil {
		c.flagged = make(map[int]flagOutcome)
	}
	c.flagged[number] = flagFailed
}

// flagAttemptFailed reports whether this run already tried and failed to flag.
func (c *GitHubClient) flagAttemptFailed(number int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.flagged[number] == flagFailed
}

// SearchSpamCandidates returns up to IssueCount open issues (most recently
// updated first) that do not already carry the spam label, optionally restricted
// to a freshness window. The window filters on update time because spam often
// arrives as a comment on an older issue. Pull requests are excluded.
func (c *GitHubClient) SearchSpamCandidates(ctx context.Context) ([]int, error) {
	query := fmt.Sprintf("repo:%s/%s is:issue state:open -label:%q", c.cfg.Owner, c.cfg.Repo, c.cfg.SpamLabel)
	if c.cfg.FreshnessWindow > 0 {
		// Full RFC3339 timestamp (not date-only) so sub-day windows keep their
		// precision: the GitHub Search API honors updated:>=YYYY-MM-DDTHH:MM:SSZ.
		cutoff := time.Now().UTC().Add(-c.cfg.FreshnessWindow).Format("2006-01-02T15:04:05Z")
		query += " updated:>=" + cutoff
	}
	c.log.Info("searching for spam candidates", "query", query)

	opts := &github.SearchOptions{
		Sort:        "updated",
		Order:       "desc",
		ListOptions: github.ListOptions{PerPage: searchPageSize},
	}
	var numbers []int
	seen := make(map[int]bool)
	for {
		result, resp, err := c.rest.Search.Issues(ctx, query, opts)
		if err != nil {
			return nil, fmt.Errorf("search issues: %w", err)
		}
		for _, issue := range result.Issues {
			if issue.IsPullRequest() {
				continue
			}
			// Dedup: an issue updated mid-pagination can appear on two pages;
			// processing the same number twice risks a duplicate alert.
			n := issue.GetNumber()
			if seen[n] {
				continue
			}
			seen[n] = true
			numbers = append(numbers, n)
			if len(numbers) >= c.cfg.IssueCount {
				return numbers, nil
			}
		}
		if resp.NextPage == 0 {
			break
		}
		opts.Page = resp.NextPage
	}
	c.log.Info("found spam candidates", "count", len(numbers))
	return numbers, nil
}

// --- GraphQL plumbing -------------------------------------------------------
//
// The raw shapes mirror the GraphQL response so the client can decode directly
// into them; toIssue maps them onto the domain Issue used for review.

type ghActor struct {
	Login string `json:"login"`
	// Typename is the GraphQL __typename ("Bot" for bot accounts). GitHub's
	// GraphQL API returns a bare bot login (e.g. "github-actions") without the
	// "[bot]" suffix that REST appends, so the type is the reliable bot signal.
	Typename string `json:"__typename"`
}

type ghComment struct {
	Author            *ghActor `json:"author"`
	AuthorAssociation string   `json:"authorAssociation"`
	Body              string   `json:"body"`
}

type rawIssue struct {
	Number            int      `json:"number"`
	Title             string   `json:"title"`
	Body              string   `json:"body"`
	Author            *ghActor `json:"author"`
	AuthorAssociation string   `json:"authorAssociation"`
	Labels            struct {
		Nodes []struct {
			Name string `json:"name"`
		} `json:"nodes"`
	} `json:"labels"`
	Comments struct {
		Nodes []ghComment `json:"nodes"`
	} `json:"comments"`
}

func login(a *ghActor) string {
	if a == nil {
		return ""
	}
	// Canonicalize a bot login to the REST "[bot]" form. GraphQL returns the bare
	// login (e.g. "github-actions"), but selfLogin is resolved via REST
	// ("github-actions[bot]") and the ignore filter matches on the "[bot]"
	// suffix; without this, bot content would slip through to the model and the
	// bot could fail to recognize its own alert comments.
	if a.Typename == "Bot" && !strings.HasSuffix(a.Login, "[bot]") {
		return a.Login + "[bot]"
	}
	return a.Login
}

func (r *rawIssue) toIssue() Issue {
	labels := make([]string, 0, len(r.Labels.Nodes))
	for _, l := range r.Labels.Nodes {
		labels = append(labels, l.Name)
	}
	comments := make([]Comment, 0, len(r.Comments.Nodes))
	for _, c := range r.Comments.Nodes {
		comments = append(comments, Comment{Author: login(c.Author), Association: c.AuthorAssociation, Body: c.Body})
	}
	return Issue{
		Number:      r.Number,
		Title:       r.Title,
		Body:        r.Body,
		Author:      login(r.Author),
		Association: r.AuthorAssociation,
		Labels:      labels,
		Comments:    comments,
	}
}

const issueQuery = `
query($owner: String!, $name: String!, $number: Int!, $commentLimit: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      number
      title
      body
      author { login __typename }
      authorAssociation
      labels(first: 100) { nodes { name } }
      comments(last: $commentLimit) {
        nodes { author { login __typename } authorAssociation body }
      }
    }
  }
}`

type issueResponse struct {
	Data struct {
		// Repository is a pointer so a null (couldn't resolve the repository,
		// e.g. wrong owner/repo or missing access) is distinguishable from a
		// resolved repository whose issue is null (issue not found / a PR).
		Repository *struct {
			Issue *rawIssue `json:"issue"`
		} `json:"repository"`
	} `json:"data"`
	Errors []struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"errors"`
}

// FetchIssue retrieves an issue and its recent comments in a single GraphQL
// query, issued through the authenticated go-github client (no extra
// dependency). It returns ErrIssueNotFound when the number does not exist or
// refers to a pull request.
func (c *GitHubClient) FetchIssue(ctx context.Context, number int) (Issue, error) {
	body := map[string]any{
		"query": issueQuery,
		"variables": map[string]any{
			"owner":  c.cfg.Owner,
			"name":   c.cfg.Repo,
			"number": number,
			// A bounded window keeps the query cheap. The spam LABEL is the
			// primary idempotency guard (the search excludes -label:spam and
			// alreadyHandled checks it); the bot's own alert comment is only a
			// best-effort secondary signal, so on a thread with more than this
			// many comments after the alert, hasBotAlert may miss it. That can
			// cause a re-alert only if the label was also removed.
			"commentLimit": 100,
		},
	}
	req, err := c.rest.NewRequest("POST", "graphql", body)
	if err != nil {
		return Issue{}, fmt.Errorf("build graphql request: %w", err)
	}
	var out issueResponse
	if _, err := c.rest.Do(ctx, req, &out); err != nil {
		return Issue{}, fmt.Errorf("graphql request: %w", err)
	}
	// A null repository means we could not resolve OWNER/REPO at all (misconfig
	// or missing access). That is an infrastructure error, NOT "issue not found":
	// mapping it to ErrIssueNotFound would make a misconfigured bot skip every
	// issue and exit 0. Surface it so the run fails loudly.
	if out.Data.Repository == nil {
		msg := "could not resolve repository"
		if len(out.Errors) > 0 {
			msg = out.Errors[0].Message
		}
		return Issue{}, fmt.Errorf("resolve repository %s/%s: %s", c.cfg.Owner, c.cfg.Repo, msg)
	}
	// Inspect GraphQL errors BEFORE treating a null issue as not-found. GitHub
	// signals a genuinely missing issue/PR with a NOT_FOUND-typed error (map that
	// to ErrIssueNotFound → skip), but a transient error (rate limit, timeout,
	// query-complexity) also returns issue:null with a DIFFERENT error type — that
	// is infrastructure failure and must fail loud, not be silently skipped.
	if len(out.Errors) > 0 {
		for _, e := range out.Errors {
			if e.Type == "NOT_FOUND" {
				return Issue{}, fmt.Errorf("issue #%d: %w", number, ErrIssueNotFound)
			}
		}
		return Issue{}, fmt.Errorf("graphql error: %s", out.Errors[0].Message)
	}
	// Repository resolved, no errors, but the issue is null: the number does not
	// exist or refers to a pull request (issue() resolves only Issues).
	if out.Data.Repository.Issue == nil {
		return Issue{}, fmt.Errorf("issue #%d: %w", number, ErrIssueNotFound)
	}
	return out.Data.Repository.Issue.toIssue(), nil
}

// --- Mutations (all honor dry-run) ------------------------------------------

// FlagSpam posts the maintainer alert, then applies the spam label. The comment
// is written first because it is the notification that actually matters: if the
// label step then fails, the next run finds the issue unlabeled, hasBotAlert
// recognizes this comment and skips it, so the comment is never duplicated.
// (Labeling first would be worse: a failed comment would leave the issue
// labeled-but-unexplained and excluded from future sweeps, so maintainers would
// never be alerted at all.) Self-recognition needs a resolved identity; with an
// unresolved selfLogin, idempotency falls back to the within-run guard plus the
// label that any earlier successful run applied.
func (c *GitHubClient) FlagSpam(ctx context.Context, number int, comment string) error {
	if err := c.postComment(ctx, number, comment); err != nil {
		return fmt.Errorf("post alert comment: %w", err)
	}
	if err := c.addLabel(ctx, number, c.cfg.SpamLabel); err != nil {
		return fmt.Errorf("add spam label: %w", err)
	}
	return nil
}

// addLabel adds a label to the issue. It is a no-op under dry-run.
func (c *GitHubClient) addLabel(ctx context.Context, number int, label string) error {
	if c.shouldSkip(number, "add label %q", label) {
		return nil
	}
	if _, _, err := c.rest.Issues.AddLabelsToIssue(ctx, c.cfg.Owner, c.cfg.Repo, number, []string{label}); err != nil {
		return err
	}
	c.log.Info("added label", "issue", number, "label", label)
	return nil
}

// postComment posts a comment on the issue. It is a no-op under dry-run.
func (c *GitHubClient) postComment(ctx context.Context, number int, body string) error {
	if c.shouldSkip(number, "post alert comment") {
		return nil
	}
	if _, _, err := c.rest.Issues.CreateComment(ctx, c.cfg.Owner, c.cfg.Repo, number, &github.IssueComment{Body: github.String(body)}); err != nil {
		return err
	}
	c.log.Info("posted alert comment", "issue", number)
	return nil
}

// shouldSkip logs an intended mutation and reports whether it should be skipped
// because dry-run is enabled. It is the single chokepoint every mutation passes
// through, so dry-run is impossible to forget.
func (c *GitHubClient) shouldSkip(number int, format string, args ...any) bool {
	if c.cfg.DryRun {
		c.log.Info("[dry-run] would "+fmt.Sprintf(format, args...), "issue", number)
		return true
	}
	return false
}
