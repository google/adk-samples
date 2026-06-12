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
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/google/go-github/v66/github"
)

func testConfig() *Config {
	return &Config{
		Owner:         "google",
		Repo:          "adk-go",
		AllowedLabels: defaultAllowedLabels,
		IssueCount:    3,
	}
}

func testClient(t *testing.T, cfg *Config, h http.Handler) *Client {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	base, err := url.Parse(srv.URL + "/")
	if err != nil {
		t.Fatalf("parse base url: %v", err)
	}
	rest := github.NewClient(nil)
	rest.BaseURL = base
	return &Client{
		rest: rest,
		cfg:  cfg,
		log:  slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
}

func TestListUntriaged(t *testing.T) {
	// Two issues need triage (no type / missing label), one is fully triaged
	// and must be filtered out.
	const body = `{"data":{"search":{"pageInfo":{"hasNextPage":false,"endCursor":""},
		"nodes":[
			{"number":10,"title":"crash on nil","body":"boom","issueType":null,"labels":{"nodes":[]}},
			{"number":11,"title":"add feature","body":"please","issueType":{"name":"Feature"},"labels":{"nodes":[{"name":"go"}]}},
			{"number":12,"title":"done","body":"x","issueType":{"name":"Bug"},"labels":{"nodes":[{"name":"bug"}]}}
		]}}}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/graphql" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		_, _ = io.WriteString(w, body)
	}))

	issues, err := c.ListUntriaged(context.Background(), 3)
	if err != nil {
		t.Fatalf("ListUntriaged() error = %v", err)
	}
	if len(issues) != 2 {
		t.Fatalf("got %d issues, want 2 (fully-triaged should be filtered): %+v", len(issues), issues)
	}
	if issues[0].Number != 10 || issues[1].Number != 11 {
		t.Errorf("unexpected issue numbers: %d, %d", issues[0].Number, issues[1].Number)
	}
}

func TestListUntriagedRespectsCount(t *testing.T) {
	const body = `{"data":{"search":{"pageInfo":{"hasNextPage":false},
		"nodes":[
			{"number":1,"issueType":null,"labels":{"nodes":[]}},
			{"number":2,"issueType":null,"labels":{"nodes":[]}},
			{"number":3,"issueType":null,"labels":{"nodes":[]}}
		]}}}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	issues, err := c.ListUntriaged(context.Background(), 2)
	if err != nil {
		t.Fatalf("ListUntriaged() error = %v", err)
	}
	if len(issues) != 2 {
		t.Fatalf("got %d issues, want 2 (count cap)", len(issues))
	}
}

func TestListUntriagedFollowsPagination(t *testing.T) {
	// Page 1 has one untriaged issue and signals a next page; the cursor must be
	// followed to page 2 to collect the second.
	page1 := `{"data":{"search":{"pageInfo":{"hasNextPage":true,"endCursor":"CURSOR2"},
		"nodes":[{"number":1,"issueType":null,"labels":{"nodes":[]}}]}}}`
	page2 := `{"data":{"search":{"pageInfo":{"hasNextPage":false,"endCursor":""},
		"nodes":[{"number":2,"issueType":null,"labels":{"nodes":[]}}]}}}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Variables struct {
				After string `json:"after"`
			} `json:"variables"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Variables.After == "" {
			_, _ = io.WriteString(w, page1)
		} else {
			_, _ = io.WriteString(w, page2)
		}
	}))
	issues, err := c.ListUntriaged(context.Background(), 5)
	if err != nil {
		t.Fatalf("ListUntriaged() error = %v", err)
	}
	if len(issues) != 2 || issues[0].Number != 1 || issues[1].Number != 2 {
		t.Fatalf("pagination not followed; got %+v", issues)
	}
}

func TestGetIssueFound(t *testing.T) {
	const body = `{"data":{"repository":{"issue":{"number":42,"title":"t","body":"b",
		"issueType":{"name":"Bug"},"labels":{"nodes":[{"name":"bug"}]}}}}}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	iss, err := c.GetIssue(context.Background(), 42)
	if err != nil {
		t.Fatalf("GetIssue() error = %v", err)
	}
	if iss.Number != 42 || iss.Type != "Bug" || len(iss.Labels) != 1 || iss.Labels[0] != "bug" {
		t.Errorf("unexpected issue: %+v", iss)
	}
}

func TestGetIssueNotFoundOrPR(t *testing.T) {
	// GraphQL returns a null issue for a non-existent number or a pull request.
	const body = `{"data":{"repository":{"issue":null}}}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	if _, err := c.GetIssue(context.Background(), 999); !errors.Is(err, ErrIssueNotFound) {
		t.Fatalf("GetIssue() error = %v, want ErrIssueNotFound", err)
	}
}

func TestGetIssueNotFoundError(t *testing.T) {
	// A PR number or non-existent issue yields a NOT_FOUND error (not a null
	// issue) from GitHub's GraphQL API.
	const body = `{"data":{"repository":{"issue":null}},"errors":[{"type":"NOT_FOUND",` +
		`"message":"Could not resolve to an Issue with the number of 1005."}]}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	if _, err := c.GetIssue(context.Background(), 1005); !errors.Is(err, ErrIssueNotFound) {
		t.Fatalf("GetIssue() error = %v, want ErrIssueNotFound", err)
	}
}

func TestGetIssueGraphQLError(t *testing.T) {
	const body = `{"errors":[{"message":"Something went wrong"}]}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	_, err := c.GetIssue(context.Background(), 1)
	if err == nil || !strings.Contains(err.Error(), "Something went wrong") {
		t.Fatalf("GetIssue() error = %v, want graphql error propagated", err)
	}
}

func TestSetTypeSendsPatch(t *testing.T) {
	var (
		gotMethod string
		gotPath   string
		gotBody   map[string]any
	)
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = io.WriteString(w, `{}`)
	}))
	if err := c.SetType(context.Background(), 7, "Bug"); err != nil {
		t.Fatalf("SetType() error = %v", err)
	}
	if gotMethod != http.MethodPatch {
		t.Errorf("method = %s, want PATCH", gotMethod)
	}
	if gotPath != "/repos/google/adk-go/issues/7" {
		t.Errorf("path = %s", gotPath)
	}
	if gotBody["type"] != "Bug" {
		t.Errorf("body type = %v, want Bug", gotBody["type"])
	}
}

func TestAddLabelHitsEndpoint(t *testing.T) {
	var gotPath string
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_, _ = io.WriteString(w, `[{"name":"bug"}]`)
	}))
	if err := c.AddLabel(context.Background(), 7, "bug"); err != nil {
		t.Fatalf("AddLabel() error = %v", err)
	}
	if gotPath != "/repos/google/adk-go/issues/7/labels" {
		t.Errorf("path = %s", gotPath)
	}
}

func TestDryRunMakesNoMutatingCalls(t *testing.T) {
	cfg := testConfig()
	cfg.DryRun = true
	var calls int
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `{}`)
	}))
	if err := c.SetType(context.Background(), 1, "Bug"); err != nil {
		t.Fatalf("SetType() dry-run error = %v", err)
	}
	if err := c.AddLabel(context.Background(), 1, "bug"); err != nil {
		t.Fatalf("AddLabel() dry-run error = %v", err)
	}
	if calls != 0 {
		t.Errorf("dry-run made %d HTTP calls, want 0", calls)
	}
}
