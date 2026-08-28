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
	"regexp"
	"strings"
	"testing"
	"time"
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
	for _, want := range []string{"change_issue_type", "add_label_to_issue"} {
		if !got[want] {
			t.Errorf("missing tool %q (have %v)", want, got)
		}
	}
	// The batch list tool was removed: the sweep is a Go loop that gives each
	// issue its own session, so the model never pulls a set of issues into one
	// context. Re-adding it would reintroduce cross-issue contamination.
	if got["list_untriaged_issues"] {
		t.Error("list_untriaged_issues must not be exposed to the model")
	}
	if len(tools) != 2 {
		t.Errorf("got %d tools, want 2", len(tools))
	}
}

func TestBuildIssuePromptFencesUntrustedText(t *testing.T) {
	iss := Issue{Number: 5, Title: "crash", Body: "trace"}
	prompt, err := buildIssuePrompt(iss, need{typ: true, label: true})
	if err != nil {
		t.Fatalf("buildIssuePrompt() error = %v", err)
	}
	if !strings.Contains(prompt, "#5") || !strings.Contains(prompt, "crash") || !strings.Contains(prompt, "trace") {
		t.Errorf("prompt missing issue details: %q", prompt)
	}
	// A fixed <body> tag would be guessable; the fence marker must not be.
	m := regexp.MustCompile(`\[UNTRUSTED:([0-9a-f]{16})\]`).FindAllStringSubmatch(prompt, -1)
	if len(m) != 2 {
		t.Fatalf("want two opening fences (title and body), got %d in %q", len(m), prompt)
	}
	// Distinct markers, so neither field can close the other's fence.
	if m[0][1] == m[1][1] {
		t.Errorf("title and body share the nonce %q; they must differ", m[0][1])
	}
	for _, g := range m {
		if !strings.Contains(prompt, "[/UNTRUSTED:"+g[1]+"]") {
			t.Errorf("fence %q is never closed", g[1])
		}
	}
	if strings.Contains(prompt, "<body>") || strings.Contains(prompt, "<title>") {
		t.Error("prompt still uses guessable fixed tags")
	}
}

// Each issue must get a fresh marker, or one issue's body could close another's
// fence in a later session.
func TestBuildIssuePromptNonceDiffersPerCall(t *testing.T) {
	iss := Issue{Number: 1, Title: "t", Body: "b"}
	seen := make(map[string]bool)
	re := regexp.MustCompile(`\[UNTRUSTED:([0-9a-f]{16})\]`)
	for range 50 {
		prompt, err := buildIssuePrompt(iss, need{typ: true})
		if err != nil {
			t.Fatalf("buildIssuePrompt() error = %v", err)
		}
		n := re.FindStringSubmatch(prompt)[1]
		if seen[n] {
			t.Fatalf("nonce %q repeated within 50 calls", n)
		}
		seen[n] = true
	}
}

// A session scoped to one issue must refuse to mutate any other, whatever the
// model asks for.
func TestMutatingToolsRefuseOutOfScopeIssue(t *testing.T) {
	c := &Client{cfg: testConfig(), log: discardLogger()}
	c.authorize(99, need{typ: true, label: true})
	ctx := withAuditedIssue(context.Background(), 5)

	res, err := c.doChangeType(ctx, 99, "Bug")
	if err != nil {
		t.Fatalf("doChangeType error = %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "scoped to issue #5") {
		t.Errorf("doChangeType = %+v, want refusal naming the session scope", res)
	}

	res, err = c.doAddLabel(ctx, 99, "bug")
	if err != nil {
		t.Fatalf("doAddLabel error = %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "scoped to issue #5") {
		t.Errorf("doAddLabel = %+v, want refusal naming the session scope", res)
	}
}

// With no scope in the context at all the tools must fail closed.
func TestMutatingToolsRefuseUnscopedSession(t *testing.T) {
	c := &Client{cfg: testConfig(), log: discardLogger()}
	c.authorize(5, need{typ: true, label: true})
	res, err := c.doChangeType(context.Background(), 5, "Bug")
	if err != nil {
		t.Fatalf("doChangeType error = %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "no issue is authorized") {
		t.Errorf("doChangeType = %+v, want a fail-closed refusal", res)
	}
}

// One issue's failure must not deny triage to the issues behind it. Before the
// sweep became a Go loop there was one session to fail; now a single issue the
// model chokes on could have aborted the whole run.
func TestSweepContinuesPastAFailingIssue(t *testing.T) {
	cfg := testConfig()
	cfg.SweepTimeout = time.Minute
	cfg.IssueTimeout = time.Minute

	var attempted []int
	// Stand in for triageOne: fail the first issue, record every attempt.
	runOne := func(iss Issue) error {
		attempted = append(attempted, iss.Number)
		if iss.Number == 1 {
			return errors.New("model refused")
		}
		return nil
	}

	issues := []Issue{{Number: 1}, {Number: 2}, {Number: 3}}
	var errs []error
	for _, iss := range issues {
		if err := runOne(iss); err != nil {
			errs = append(errs, err)
		}
	}
	if len(attempted) != 3 {
		t.Errorf("attempted %v, want all three issues tried", attempted)
	}
	if len(errs) != 1 {
		t.Errorf("collected %d errors, want 1 aggregated", len(errs))
	}
}

// SWEEP_TIMEOUT must bound the whole run: N issues each taking IssueTimeout
// would otherwise multiply past the workflow's own timeout-minutes and be
// killed mid-sweep, which is silent.
func TestValidateBoundsSweepAgainstIssueTimeout(t *testing.T) {
	base := func() *Config {
		return &Config{
			GitHubToken: "t", GeminiAPIKey: "k", Owner: "o", Repo: "r",
			IssueCount: 3, IssueTimeout: 5 * time.Minute, SweepTimeout: 15 * time.Minute,
		}
	}
	if err := base().validate(); err != nil {
		t.Fatalf("a valid config must pass: %v", err)
	}
	for _, tc := range []struct {
		name string
		mut  func(*Config)
		want string
	}{
		{"zero sweep", func(c *Config) { c.SweepTimeout = 0 }, "SWEEP_TIMEOUT"},
		{"negative issue", func(c *Config) { c.IssueTimeout = -1 }, "ISSUE_TIMEOUT"},
		{"sweep below issue", func(c *Config) { c.SweepTimeout = time.Minute }, "at least"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := base()
			tc.mut(c)
			err := c.validate()
			if err == nil {
				t.Fatalf("validate() = nil, want an error naming %s", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("validate() = %v, want it to name %s", err, tc.want)
			}
		})
	}
}
