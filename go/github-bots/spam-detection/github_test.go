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
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/google/go-github/v66/github"
)

func testConfig() *Config {
	return &Config{
		Owner:       "google",
		Repo:        "adk-go",
		SpamLabel:   "spam",
		IssueCount:  3,
		Concurrency: 1,
	}
}

// respondWith builds a client whose server answers every request with one
// canned body. Most tests only need that, and spelling out an http.HandlerFunc
// at each call site buried the body being asserted on.
func respondWith(t *testing.T, cfg *Config, body string) *GitHubClient {
	t.Helper()
	return testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
}

func testClient(t *testing.T, cfg *Config, h http.Handler) *GitHubClient {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	base, err := url.Parse(srv.URL + "/")
	if err != nil {
		t.Fatalf("parse base url: %v", err)
	}
	rest := github.NewClient(nil)
	rest.BaseURL = base
	return &GitHubClient{
		rest:      rest,
		cfg:       cfg,
		selfLogin: "spam-bot",
		log:       slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
}

func TestFetchIssueGraphQLErrorFailsLoud(t *testing.T) {
	// A transient GraphQL error returns issue:null WITH an error. It must surface
	// as a real error (fail loud), not be masked as ErrIssueNotFound (which would
	// silently skip the issue and exit 0).
	const body = `{"data":{"repository":{"issue":null}},"errors":[{"message":"rate limited"}]}`
	c := respondWith(t, testConfig(), body)
	_, err := c.FetchIssue(context.Background(), 5)
	if err == nil || errors.Is(err, ErrIssueNotFound) || !strings.Contains(err.Error(), "rate limited") {
		t.Fatalf("FetchIssue() error = %v, want a graphql error (not ErrIssueNotFound)", err)
	}
}

func TestFetchIssueCanonicalizesBotLogin(t *testing.T) {
	// GraphQL returns a bare bot login ("github-actions"); toIssue must canonicalize
	// it to the REST "[bot]" form so the ignore filter and self-identity match.
	const body = `{"data":{"repository":{"issue":{
		"number":5,"title":"t","body":"b",
		"author":{"login":"alice","__typename":"User"},
		"authorAssociation":"NONE","labels":{"nodes":[]},
		"comments":{"nodes":[
			{"author":{"login":"github-actions","__typename":"Bot"},"authorAssociation":"NONE","body":"beep"}
		]}}}}}`
	c := respondWith(t, testConfig(), body)
	iss, err := c.FetchIssue(context.Background(), 5)
	if err != nil {
		t.Fatalf("FetchIssue() error = %v", err)
	}
	if len(iss.Comments) != 1 || iss.Comments[0].Author != "github-actions[bot]" {
		t.Errorf("bot comment author = %q, want canonical %q", iss.Comments[0].Author, "github-actions[bot]")
	}
}

func TestSearchSpamCandidatesExcludesPRs(t *testing.T) {
	const body = `{"total_count":3,"incomplete_results":false,"items":[
		{"number":1},
		{"number":2,"pull_request":{"url":"https://api.github.com/repos/google/adk-go/pulls/2"}},
		{"number":3}
	]}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/search/issues" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		_, _ = io.WriteString(w, body)
	}))
	got, err := c.SearchSpamCandidates(context.Background())
	if err != nil {
		t.Fatalf("SearchSpamCandidates() error = %v", err)
	}
	want := []int{1, 3}
	if len(got) != len(want) || got[0] != 1 || got[1] != 3 {
		t.Errorf("SearchSpamCandidates() = %v, want %v (PR excluded)", got, want)
	}
}

func TestSearchSpamCandidatesRespectsCount(t *testing.T) {
	const body = `{"total_count":3,"incomplete_results":false,"items":[
		{"number":1},{"number":2},{"number":3}
	]}`
	cfg := testConfig()
	cfg.IssueCount = 1
	c := respondWith(t, cfg, body)
	got, err := c.SearchSpamCandidates(context.Background())
	if err != nil {
		t.Fatalf("SearchSpamCandidates() error = %v", err)
	}
	if len(got) != 1 || got[0] != 1 {
		t.Errorf("SearchSpamCandidates() = %v, want [1] (count cap)", got)
	}
}

func TestSearchSpamCandidatesQueryExcludesLabelAndFreshness(t *testing.T) {
	cfg := testConfig()
	cfg.FreshnessWindow = 24 * time.Hour
	var gotQuery string
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.Query().Get("q")
		_, _ = io.WriteString(w, `{"items":[]}`)
	}))
	if _, err := c.SearchSpamCandidates(context.Background()); err != nil {
		t.Fatalf("SearchSpamCandidates() error = %v", err)
	}
	if !strings.Contains(gotQuery, `-label:"spam"`) {
		t.Errorf("query %q missing -label:\"spam\"", gotQuery)
	}
	if !strings.Contains(gotQuery, "updated:>=") {
		t.Errorf("query %q missing freshness filter", gotQuery)
	}
	// Full RFC3339 timestamp, not date-only, so sub-day windows keep precision.
	if !strings.Contains(gotQuery, "T") || !strings.Contains(gotQuery, "Z") {
		t.Errorf("query %q freshness cutoff is not a full datetime", gotQuery)
	}
}

func TestFetchIssueFound(t *testing.T) {
	const body = `{"data":{"repository":{"issue":{
		"number":42,"title":"t","body":"b","author":{"login":"alice"},"authorAssociation":"FIRST_TIME_CONTRIBUTOR",
		"labels":{"nodes":[{"name":"bug"}]},
		"comments":{"nodes":[{"author":{"login":"bob"},"authorAssociation":"NONE","body":"hi"},{"author":null,"body":"ghost"}]}
	}}}}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/graphql" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		_, _ = io.WriteString(w, body)
	}))
	iss, err := c.FetchIssue(context.Background(), 42)
	if err != nil {
		t.Fatalf("FetchIssue() error = %v", err)
	}
	if iss.Number != 42 || iss.Author != "alice" || iss.Title != "t" || iss.Body != "b" {
		t.Errorf("unexpected issue: %+v", iss)
	}
	if iss.Association != "FIRST_TIME_CONTRIBUTOR" {
		t.Errorf("issue association = %q, want FIRST_TIME_CONTRIBUTOR", iss.Association)
	}
	if len(iss.Labels) != 1 || iss.Labels[0] != "bug" {
		t.Errorf("labels = %v, want [bug]", iss.Labels)
	}
	if len(iss.Comments) != 2 || iss.Comments[0].Author != "bob" || iss.Comments[1].Author != "" {
		t.Errorf("comments = %+v (want bob + empty-author ghost)", iss.Comments)
	}
	if iss.Comments[0].Association != "NONE" {
		t.Errorf("comment association = %q, want NONE", iss.Comments[0].Association)
	}
}

func TestFetchIssueNotFoundNull(t *testing.T) {
	const body = `{"data":{"repository":{"issue":null}}}`
	c := respondWith(t, testConfig(), body)
	if _, err := c.FetchIssue(context.Background(), 999); !errors.Is(err, ErrIssueNotFound) {
		t.Fatalf("FetchIssue() error = %v, want ErrIssueNotFound", err)
	}
}

func TestFetchIssueNotFoundError(t *testing.T) {
	const body = `{"data":{"repository":{"issue":null}},"errors":[{"type":"NOT_FOUND",` +
		`"message":"Could not resolve to an Issue with the number of 1005."}]}`
	c := respondWith(t, testConfig(), body)
	if _, err := c.FetchIssue(context.Background(), 1005); !errors.Is(err, ErrIssueNotFound) {
		t.Fatalf("FetchIssue() error = %v, want ErrIssueNotFound", err)
	}
}

func TestFetchIssueRepoNotFoundIsRealError(t *testing.T) {
	// A null repository (wrong OWNER/REPO or missing access) must surface as a
	// real error, NOT ErrIssueNotFound, so a misconfigured bot fails loudly
	// instead of silently skipping every issue and exiting 0.
	const body = `{"data":{"repository":null},"errors":[{"message":` +
		`"Could not resolve to a Repository with the name 'google/nope'."}]}`
	c := respondWith(t, testConfig(), body)
	_, err := c.FetchIssue(context.Background(), 1)
	if err == nil {
		t.Fatal("FetchIssue() on null repository expected error, got nil")
	}
	if errors.Is(err, ErrIssueNotFound) {
		t.Errorf("FetchIssue() on null repository = ErrIssueNotFound, want a real (fail-loud) error: %v", err)
	}
	if !strings.Contains(err.Error(), "Could not resolve to a Repository") {
		t.Errorf("error %v should carry the underlying GraphQL message", err)
	}
}

func TestFetchIssueGraphQLError(t *testing.T) {
	const body = `{"errors":[{"message":"Something went wrong"}]}`
	c := respondWith(t, testConfig(), body)
	_, err := c.FetchIssue(context.Background(), 1)
	if err == nil || !strings.Contains(err.Error(), "Something went wrong") {
		t.Fatalf("FetchIssue() error = %v, want graphql error propagated", err)
	}
}

func TestFlagSpamCommentsThenLabels(t *testing.T) {
	var paths []string
	var commentBody string
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		if strings.HasSuffix(r.URL.Path, "/comments") {
			b, _ := io.ReadAll(r.Body)
			commentBody = string(b)
			_, _ = io.WriteString(w, `{"id":1}`)
			return
		}
		_, _ = io.WriteString(w, `[{"name":"spam"}]`)
	}))
	if err := c.FlagSpam(context.Background(), 7, buildAlertComment("promo link")); err != nil {
		t.Fatalf("FlagSpam() error = %v", err)
	}
	if len(paths) != 2 {
		t.Fatalf("made %d calls, want 2 (comment + label): %v", len(paths), paths)
	}
	// The alert comment must be posted before the label: it is the notification,
	// and posting it first keeps a failed label from silently dropping the alert.
	if !strings.HasSuffix(paths[0], "/issues/7/comments") {
		t.Errorf("first call = %s, want comments endpoint", paths[0])
	}
	if !strings.HasSuffix(paths[1], "/issues/7/labels") {
		t.Errorf("second call = %s, want labels endpoint", paths[1])
	}
	if !strings.Contains(commentBody, "Automated spam detection") {
		t.Errorf("comment body missing signature: %s", commentBody)
	}
}

func TestFlagSpamDryRunMakesNoCalls(t *testing.T) {
	cfg := testConfig()
	cfg.DryRun = true
	var calls int
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `{}`)
	}))
	if err := c.FlagSpam(context.Background(), 1, buildAlertComment("x")); err != nil {
		t.Fatalf("FlagSpam() dry-run error = %v", err)
	}
	if calls != 0 {
		t.Errorf("dry-run made %d HTTP calls, want 0", calls)
	}
}

// "never review a maintainer's comment" rests entirely on this one wiring line.
// Every other test passes its own maintainerSet, so deleting the assignment in
// NewGitHubClient left a nil map and the whole invariant silently off.
func TestNewGitHubClientWiresTheMaintainerSet(t *testing.T) {
	// No network: NewGitHubClient builds its own REST client, so an httptest
	// server cannot be injected and the identity lookup would hit the real
	// api.github.com. Cancel the context so that call fails fast -- it is
	// best-effort by design -- and assert on the wiring, which is what this test
	// is about.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	cfg := testConfig()
	cfg.Maintainers = []string{"Wolo-Lab", "dpasiukevich"}
	c, err := NewGitHubClient(ctx, cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("NewGitHubClient: %v", err)
	}
	for _, login := range []string{"wolo-lab", "WOLO-LAB", "dpasiukevich"} {
		if !c.maintainers[strings.ToLower(login)] {
			t.Errorf("maintainer %q is not in the set; their comments would be reviewed as spam", login)
		}
	}
}
