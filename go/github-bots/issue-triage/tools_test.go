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
	"testing"

	"github.com/google/go-github/v66/github"
)

// countingClient returns a client whose handler counts every HTTP call, so
// tests can assert that rejected actions make no network calls.
func countingClient(t *testing.T, cfg *Config, status int, body string) (*Client, *int) {
	t.Helper()
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	base, _ := url.Parse(srv.URL + "/")
	rest := github.NewClient(nil)
	rest.BaseURL = base
	return &Client{
		rest:       rest,
		cfg:        cfg,
		log:        slog.New(slog.NewTextHandler(io.Discard, nil)),
		authorized: make(map[int]need),
	}, &calls
}

func TestDoChangeTypeRejectsWithoutHTTP(t *testing.T) {
	tests := []struct {
		name       string
		number     int
		issueType  string
		authorize  bool
		wantStatus string
	}{
		{"disallowed type", 7, "Epic", true, "error"},
		{"unauthorized issue", 7, "Bug", false, "error"},
		{"invalid number", 0, "Bug", true, "error"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c, calls := countingClient(t, testConfig(), http.StatusOK, `{}`)
			if tc.authorize {
				c.authorize(tc.number, need{typ: true})
			}
			res, err := c.doChangeType(context.Background(), tc.number, tc.issueType)
			if err != nil {
				t.Fatalf("doChangeType() unexpected Go error = %v", err)
			}
			if res.Status != tc.wantStatus {
				t.Errorf("status = %q, want %q", res.Status, tc.wantStatus)
			}
			if *calls != 0 {
				t.Errorf("made %d HTTP calls, want 0", *calls)
			}
		})
	}
}

func TestDoChangeTypeAuthorizedSucceeds(t *testing.T) {
	c, calls := countingClient(t, testConfig(), http.StatusOK, `{}`)
	c.authorize(7, need{typ: true})
	res, err := c.doChangeType(context.Background(), 7, "Bug")
	if err != nil {
		t.Fatalf("doChangeType() error = %v", err)
	}
	if res.Status != "success" {
		t.Errorf("status = %q, want success", res.Status)
	}
	if *calls != 1 {
		t.Errorf("made %d HTTP calls, want 1", *calls)
	}
}

func TestDoAddLabelRejectsWithoutHTTP(t *testing.T) {
	tests := []struct {
		name      string
		number    int
		label     string
		authorize bool
	}{
		{"disallowed label", 7, "good first issue", true},
		{"unauthorized issue", 7, "bug", false},
		{"invalid number", 0, "bug", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c, calls := countingClient(t, testConfig(), http.StatusOK, `[]`)
			if tc.authorize {
				c.authorize(tc.number, need{label: true})
			}
			res, err := c.doAddLabel(context.Background(), tc.number, tc.label)
			if err != nil {
				t.Fatalf("doAddLabel() unexpected Go error = %v", err)
			}
			if res.Status != "error" {
				t.Errorf("status = %q, want error", res.Status)
			}
			if *calls != 0 {
				t.Errorf("made %d HTTP calls, want 0", *calls)
			}
		})
	}
}

func TestDoChangeTypeRESTErrorIsGoError(t *testing.T) {
	// Infrastructure failures (non-2xx) must surface as a Go error (not an
	// errResult) AND be recorded so the run fails loudly.
	c, _ := countingClient(t, testConfig(), http.StatusInternalServerError, `{"message":"boom"}`)
	c.authorize(7, need{typ: true})
	if c.hadToolError() {
		t.Fatal("hadToolError() should start false")
	}
	if _, err := c.doChangeType(context.Background(), 7, "Bug"); err == nil {
		t.Fatal("doChangeType() expected Go error on HTTP 500, got nil")
	}
	if !c.hadToolError() {
		t.Error("doChangeType() did not record the infrastructure error")
	}
}

func TestDoChangeTypeConsumesNeed(t *testing.T) {
	// After a successful type set, a second set on the same issue this run must
	// be refused (don't overwrite what we just set) without an HTTP call.
	var calls int
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `{}`)
	}))
	c.authorize(7, need{typ: true})
	if res, err := c.doChangeType(context.Background(), 7, "Bug"); err != nil || res.Status != "success" {
		t.Fatalf("first doChangeType() = (%+v, %v), want success", res, err)
	}
	res, err := c.doChangeType(context.Background(), 7, "Feature")
	if err != nil {
		t.Fatalf("second doChangeType() error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("second set status = %q, want error (need already consumed)", res.Status)
	}
	if calls != 1 {
		t.Errorf("made %d HTTP calls, want 1 (second set must not call the API)", calls)
	}
}

func TestDoAddLabelConsumesNeed(t *testing.T) {
	var calls int
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `[{"name":"bug"}]`)
	}))
	c.authorize(7, need{label: true})
	if res, err := c.doAddLabel(context.Background(), 7, "bug"); err != nil || res.Status != "success" {
		t.Fatalf("first doAddLabel() = (%+v, %v), want success", res, err)
	}
	res, err := c.doAddLabel(context.Background(), 7, "enhancement")
	if err != nil {
		t.Fatalf("second doAddLabel() error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("second add status = %q, want error (need already consumed)", res.Status)
	}
	if calls != 1 {
		t.Errorf("made %d HTTP calls, want 1 (second add must not call the API)", calls)
	}
}

func TestDoChangeTypeCanonicalizesCasing(t *testing.T) {
	// The model may emit a lowercase type; GitHub must still receive the
	// canonical "Bug".
	var gotType any
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotType = body["type"]
		_, _ = io.WriteString(w, `{}`)
	}))
	c.authorize(7, need{typ: true})
	res, err := c.doChangeType(context.Background(), 7, "bug")
	if err != nil {
		t.Fatalf("doChangeType() error = %v", err)
	}
	if res.Status != "success" {
		t.Fatalf("status = %q, want success", res.Status)
	}
	if gotType != "Bug" {
		t.Errorf("GitHub received type %v, want canonical \"Bug\"", gotType)
	}
}

func TestDoAddLabelCanonicalizesCasing(t *testing.T) {
	// The model may emit a differently-cased label; GitHub must receive the
	// allowlist's exact spelling so it matches an existing label.
	var gotBody string
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = io.WriteString(w, `[{"name":"bug"}]`)
	}))
	c.authorize(7, need{label: true})
	res, err := c.doAddLabel(context.Background(), 7, "BUG")
	if err != nil {
		t.Fatalf("doAddLabel() error = %v", err)
	}
	if res.Status != "success" {
		t.Fatalf("status = %q, want success", res.Status)
	}
	if !strings.Contains(gotBody, `"bug"`) || strings.Contains(gotBody, `"BUG"`) {
		t.Errorf("GitHub received label body %q, want canonical \"bug\"", gotBody)
	}
}

func TestDoListAuthorizesReturnedIssues(t *testing.T) {
	const body = `{"data":{"search":{"pageInfo":{"hasNextPage":false},
		"nodes":[
			{"number":10,"issueType":null,"labels":{"nodes":[]}},
			{"number":11,"issueType":null,"labels":{"nodes":[]}}
		]}}}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	res, err := c.doList(context.Background(), 5)
	if err != nil {
		t.Fatalf("doList() error = %v", err)
	}
	if len(res.Issues) != 2 {
		t.Fatalf("got %d issues, want 2", len(res.Issues))
	}
	if _, ok := c.authorizedNeed(10); !ok {
		t.Error("doList did not authorize issue 10")
	}
	if _, ok := c.authorizedNeed(11); !ok {
		t.Error("doList did not authorize issue 11")
	}
	if _, ok := c.authorizedNeed(999); ok {
		t.Error("an issue that was never listed must not be authorized")
	}
}

func TestDoChangeTypeDoesNotOverwriteExistingField(t *testing.T) {
	// An issue can appear in the triage set because it needs a *label* while
	// already having a type. The type tool must refuse to overwrite it, in code.
	c, calls := countingClient(t, testConfig(), http.StatusOK, `{}`)
	c.authorize(7, need{label: true}) // type already set
	res, err := c.doChangeType(context.Background(), 7, "Bug")
	if err != nil {
		t.Fatalf("doChangeType() error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("status = %q, want error (must not overwrite existing type)", res.Status)
	}
	if *calls != 0 {
		t.Errorf("made %d HTTP calls, want 0", *calls)
	}
}

func TestDoAddLabelDoesNotOverwriteExistingField(t *testing.T) {
	c, calls := countingClient(t, testConfig(), http.StatusOK, `[]`)
	c.authorize(7, need{typ: true}) // label already set
	res, err := c.doAddLabel(context.Background(), 7, "bug")
	if err != nil {
		t.Fatalf("doAddLabel() error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("status = %q, want error (must not add a second label)", res.Status)
	}
	if *calls != 0 {
		t.Errorf("made %d HTTP calls, want 0", *calls)
	}
}
