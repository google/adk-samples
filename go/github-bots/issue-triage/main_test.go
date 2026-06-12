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
	"io"
	"log/slog"
	"net/http"
	"strings"
	"testing"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func TestTools(t *testing.T) {
	c := &Client{cfg: testConfig(), log: discardLogger()}
	tools, err := c.tools()
	if err != nil {
		t.Fatalf("tools() error = %v", err)
	}
	got := make(map[string]bool)
	for _, tl := range tools {
		got[tl.Name()] = true
	}
	for _, want := range []string{"list_untriaged_issues", "change_issue_type", "add_label_to_issue"} {
		if !got[want] {
			t.Errorf("missing tool %q (have %v)", want, got)
		}
	}
	if len(tools) != 3 {
		t.Errorf("got %d tools, want 3", len(tools))
	}
}

func TestBuildPromptBatch(t *testing.T) {
	cfg := testConfig()
	prompt, err := buildPrompt(context.Background(), nil, cfg, discardLogger())
	if err != nil {
		t.Fatalf("buildPrompt() error = %v", err)
	}
	if !strings.Contains(prompt, "list_untriaged_issues") {
		t.Errorf("batch prompt missing tool reference: %q", prompt)
	}
}

func TestBuildPromptSingleNeedsTriage(t *testing.T) {
	cfg := testConfig()
	cfg.SingleIssue = 5
	const body = `{"data":{"repository":{"issue":{"number":5,"title":"crash","body":"trace",
		"issueType":null,"labels":{"nodes":[]}}}}}`
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	prompt, err := buildPrompt(context.Background(), c, cfg, discardLogger())
	if err != nil {
		t.Fatalf("buildPrompt() error = %v", err)
	}
	if !strings.Contains(prompt, "#5") || !strings.Contains(prompt, "crash") {
		t.Errorf("single prompt missing issue details: %q", prompt)
	}
}

func TestBuildPromptSingleAlreadyTriaged(t *testing.T) {
	cfg := testConfig()
	cfg.SingleIssue = 6
	const body = `{"data":{"repository":{"issue":{"number":6,"title":"done","body":"x",
		"issueType":{"name":"Bug"},"labels":{"nodes":[{"name":"bug"}]}}}}}`
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	prompt, err := buildPrompt(context.Background(), c, cfg, discardLogger())
	if err != nil {
		t.Fatalf("buildPrompt() error = %v", err)
	}
	if prompt != "" {
		t.Errorf("expected empty prompt for already-triaged issue, got %q", prompt)
	}
}

func TestBuildPromptSingleNotFound(t *testing.T) {
	cfg := testConfig()
	cfg.SingleIssue = 7
	const body = `{"data":{"repository":{"issue":null}}}`
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	prompt, err := buildPrompt(context.Background(), c, cfg, discardLogger())
	if err != nil {
		t.Fatalf("buildPrompt() error = %v", err)
	}
	if prompt != "" {
		t.Errorf("expected empty prompt for missing issue, got %q", prompt)
	}
}
