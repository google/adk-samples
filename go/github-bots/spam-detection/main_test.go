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
	"strings"
	"testing"
	"unicode/utf8"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func TestTools(t *testing.T) {
	c := &GitHubClient{cfg: testConfig(), log: discardLogger()}
	tools, err := c.tools()
	if err != nil {
		t.Fatalf("tools() error = %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("got %d tools, want 1", len(tools))
	}
	if tools[0].Name() != "flag_issue_as_spam" {
		t.Errorf("tool name = %q, want flag_issue_as_spam", tools[0].Name())
	}
}

func TestCandidateIssuesSingle(t *testing.T) {
	cfg := testConfig()
	cfg.SingleIssue = 5
	// The single-issue path must not touch the client (nil here proves it).
	got, err := candidateIssues(context.Background(), nil, cfg)
	if err != nil {
		t.Fatalf("candidateIssues() error = %v", err)
	}
	if len(got) != 1 || got[0] != 5 {
		t.Errorf("candidateIssues() = %v, want [5]", got)
	}
}

func TestMaintainersWarning(t *testing.T) {
	if w := maintainersWarning(&Config{}); w == "" {
		t.Error("expected a warning when MAINTAINERS is empty")
	}
	if w := maintainersWarning(&Config{Maintainers: []string{"alice"}}); w != "" {
		t.Errorf("expected no warning when maintainers are set, got %q", w)
	}
}

func TestNewNonce(t *testing.T) {
	a, err := newNonce()
	if err != nil {
		t.Fatalf("newNonce() error = %v", err)
	}
	b, err := newNonce()
	if err != nil {
		t.Fatalf("newNonce() error = %v", err)
	}
	if len(a) != 16 {
		t.Errorf("nonce length = %d, want 16 hex chars", len(a))
	}
	if a == b {
		t.Errorf("two nonces were identical (%q); fence would be predictable", a)
	}
}

func TestSummarize(t *testing.T) {
	if got := summarize("line one\nline two"); got != "line one line two" {
		t.Errorf("summarize() = %q, want newlines collapsed", got)
	}
	long := strings.Repeat("x", 250)
	got := summarize(long)
	if len([]rune(got)) != 203 || !strings.HasSuffix(got, "...") {
		t.Errorf("summarize() rune length = %d, want 203 ending in ellipsis", len([]rune(got)))
	}
	// Truncation must be rune-safe: a multibyte rune at the boundary must not be
	// split into invalid UTF-8.
	multibyte := strings.Repeat("界", 250)
	got = summarize(multibyte)
	if !utf8.ValidString(got) {
		t.Errorf("summarize() produced invalid UTF-8: %q", got)
	}
	if len([]rune(got)) != 203 {
		t.Errorf("summarize() rune length = %d, want 203", len([]rune(got)))
	}
}
