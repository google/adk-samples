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
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-github/v66/github"
)

// testClient builds a GitHubClient whose REST/GraphQL calls are directed at the
// given test handler. Identity resolution is skipped (selfLogin is set
// directly) so no network round-trip is needed.
func testClient(t *testing.T, cfg *Config, handler http.Handler) *GitHubClient {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	base, err := url.Parse(srv.URL + "/")
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}
	rest := github.NewClient(nil)
	rest.BaseURL = base
	return &GitHubClient{
		rest:      rest,
		cfg:       cfg,
		selfLogin: "stale-bot[bot]",
		log:       slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
}

func baseCfg() *Config {
	return &Config{
		Owner:      "o",
		Repo:       "r",
		StaleLabel: "stale",
		StaleAfter: 168 * time.Hour,
		CloseAfter: 168 * time.Hour,
	}
}

func TestSearchOldOpenIssues_ExcludesPRs(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/search/issues" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"total_count":2,"incomplete_results":false,"items":[
			{"number":1},
			{"number":2,"pull_request":{"url":"https://example/pulls/2"}}
		]}`)
	})
	c := testClient(t, baseCfg(), handler)

	got, err := c.SearchOldOpenIssues(context.Background())
	if err != nil {
		t.Fatalf("SearchOldOpenIssues: %v", err)
	}
	if len(got) != 1 || got[0] != 1 {
		t.Errorf("got %v, want [1] (PR #2 excluded)", got)
	}
}

func TestFetchIssueHistory_DecodesGraphQL(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/graphql" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":{"repository":{"issue":{
			"author":{"login":"reporter"},
			"createdAt":"2026-01-01T00:00:00Z",
			"labels":{"nodes":[{"name":"bug"}]},
			"comments":{"nodes":[{"author":{"login":"maintainerA"},"body":"hi","createdAt":"2026-01-02T00:00:00Z","lastEditedAt":null}]},
			"userContentEdits":{"nodes":[]},
			"timelineItems":{"nodes":[]}
		}}}}`)
	})
	c := testClient(t, baseCfg(), handler)

	raw, err := c.FetchIssueHistory(context.Background(), 42)
	if err != nil {
		t.Fatalf("FetchIssueHistory: %v", err)
	}
	if raw.Author.Login != "reporter" {
		t.Errorf("author = %q, want reporter", raw.Author.Login)
	}
	if len(raw.Comments.Nodes) != 1 || raw.Comments.Nodes[0].Body != "hi" {
		t.Errorf("comments not decoded: %+v", raw.Comments.Nodes)
	}
}

func TestFetchIssueHistory_GraphQLError(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"errors":[{"message":"Some error"}]}`)
	})
	c := testClient(t, baseCfg(), handler)
	if _, err := c.FetchIssueHistory(context.Background(), 1); err == nil {
		t.Fatal("expected error from GraphQL errors array, got nil")
	}
}

func TestDryRunSuppressesMutations(t *testing.T) {
	var calls int
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	})
	cfg := baseCfg()
	cfg.DryRun = true
	c := testClient(t, cfg, handler)
	ctx := context.Background()

	if err := c.MarkStale(ctx, 7, "stale"); err != nil {
		t.Fatalf("MarkStale: %v", err)
	}
	if err := c.CloseAsStale(ctx, 7, "closing"); err != nil {
		t.Fatalf("CloseAsStale: %v", err)
	}
	if err := c.AddLabel(ctx, 7, "x"); err != nil {
		t.Fatalf("AddLabel: %v", err)
	}
	if calls != 0 {
		t.Errorf("dry-run made %d HTTP calls, want 0", calls)
	}
}

func TestMarkStaleAddsLabelBeforeComment(t *testing.T) {
	var (
		mu    sync.Mutex
		order []string
	)
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/labels"):
			order = append(order, "label")
			mu.Unlock()
			_, _ = io.WriteString(w, `[]`)
			return
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/comments"):
			order = append(order, "comment")
		}
		mu.Unlock()
		_, _ = io.WriteString(w, `{}`)
	})
	c := testClient(t, baseCfg(), handler)

	if err := c.MarkStale(context.Background(), 7, "stale body"); err != nil {
		t.Fatalf("MarkStale: %v", err)
	}
	if len(order) != 2 || order[0] != "label" || order[1] != "comment" {
		t.Errorf("call order = %v, want [label comment]", order)
	}
}

// CloseAsStale must close the issue BEFORE posting the comment: closing is
// idempotent and removes the issue from the next run's open-issue search, so a
// failed comment can't produce a duplicate closing comment on retry.
func TestCloseAsStaleClosesBeforeComment(t *testing.T) {
	var order []string
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPatch:
			order = append(order, "close")
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/comments"):
			order = append(order, "comment")
		}
		_, _ = io.WriteString(w, `{}`)
	})
	c := testClient(t, baseCfg(), handler)

	if err := c.CloseAsStale(context.Background(), 7, "closing"); err != nil {
		t.Fatalf("CloseAsStale: %v", err)
	}
	if len(order) != 2 || order[0] != "close" || order[1] != "comment" {
		t.Errorf("call order = %v, want [close comment]", order)
	}
}

// FetchIssueHistory must send the issue number and the configured node limits as
// GraphQL variables, and request description edits (the silent-edit feature).
func TestFetchIssueHistorySendsVariablesAndLimits(t *testing.T) {
	var body struct {
		Query     string         `json:"query"`
		Variables map[string]any `json:"variables"`
	}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `{"data":{"repository":{"issue":{"author":{"login":"x"},"createdAt":"2026-01-01T00:00:00Z"}}}}`)
	})
	c := testClient(t, baseCfg(), handler)

	if _, err := c.FetchIssueHistory(context.Background(), 42); err != nil {
		t.Fatalf("FetchIssueHistory: %v", err)
	}
	if got, _ := body.Variables["number"].(float64); got != 42 {
		t.Errorf("number variable = %v, want 42", body.Variables["number"])
	}
	if got, _ := body.Variables["commentLimit"].(float64); got != 50 {
		t.Errorf("commentLimit variable = %v, want 50", body.Variables["commentLimit"])
	}
	if got, _ := body.Variables["timelineLimit"].(float64); got != 50 {
		t.Errorf("timelineLimit variable = %v, want 50", body.Variables["timelineLimit"])
	}
	if !strings.Contains(body.Query, "userContentEdits") {
		t.Error("query does not request userContentEdits (silent-edit detection)")
	}
}

func TestCloseAsStaleSetsNotPlanned(t *testing.T) {
	var gotState, gotReason string
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPatch {
			var req github.IssueRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("decode patch body: %v", err)
			}
			gotState = req.GetState()
			gotReason = req.GetStateReason()
		}
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	})
	c := testClient(t, baseCfg(), handler)

	if err := c.CloseAsStale(context.Background(), 7, "closing"); err != nil {
		t.Fatalf("CloseAsStale: %v", err)
	}
	if gotState != "closed" || gotReason != "not_planned" {
		t.Errorf("PATCH state=%q reason=%q, want closed/not_planned", gotState, gotReason)
	}
}
