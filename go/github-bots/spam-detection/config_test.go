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
	"reflect"
	"testing"
	"time"
)

// setRequired sets the minimum credentials and clears optional env vars so
// defaults are observable.
func setRequired(t *testing.T) {
	t.Helper()
	t.Setenv("GITHUB_TOKEN", "test-token")
	t.Setenv("GEMINI_API_KEY", "test-key")
	// OWNER/REPO are required (no default), so they are part of the minimum.
	t.Setenv("OWNER", "google")
	t.Setenv("REPO", "adk-go")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
	for _, k := range []string{
		"LLM_MODEL_NAME", "SPAM_LABEL_NAME", "MAINTAINERS",
		"ISSUE_COUNT", "FRESHNESS_WINDOW_DAYS", "CONCURRENCY_LIMIT", "ISSUE_TIMEOUT", "DRY_RUN",
	} {
		t.Setenv(k, "")
	}
}

func TestLoadConfigDefaults(t *testing.T) {
	setRequired(t)
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig() error = %v", err)
	}
	if cfg.Owner != "google" || cfg.Repo != "adk-go" {
		t.Errorf("default owner/repo = %s/%s, want google/adk-go", cfg.Owner, cfg.Repo)
	}
	if cfg.Model != "gemini-flash-latest" {
		t.Errorf("default model = %q, want gemini-flash-latest", cfg.Model)
	}
	if cfg.SpamLabel != "spam" {
		t.Errorf("default SpamLabel = %q, want spam", cfg.SpamLabel)
	}
	if cfg.IssueCount != 3 {
		t.Errorf("default IssueCount = %d, want 3", cfg.IssueCount)
	}
	if cfg.Concurrency != 3 {
		t.Errorf("default Concurrency = %d, want 3", cfg.Concurrency)
	}
	if cfg.FreshnessWindow != 0 {
		t.Errorf("default FreshnessWindow = %v, want 0", cfg.FreshnessWindow)
	}
	if cfg.IssueTimeout != 5*time.Minute {
		t.Errorf("default IssueTimeout = %v, want 5m", cfg.IssueTimeout)
	}
	if cfg.DryRun {
		t.Error("default DryRun = true, want false")
	}
	if len(cfg.Maintainers) != 0 {
		t.Errorf("default Maintainers = %v, want empty", cfg.Maintainers)
	}
}

func TestLoadConfigMissingOwnerRepo(t *testing.T) {
	for _, unset := range []string{"OWNER", "REPO"} {
		t.Run(unset, func(t *testing.T) {
			setRequired(t)
			t.Setenv(unset, "")
			if _, err := loadConfig(nil); err == nil {
				t.Fatalf("loadConfig() with empty %s = nil error, want error (no silent default)", unset)
			}
		})
	}
}

func TestLoadConfigFlagsAndEnv(t *testing.T) {
	setRequired(t)
	t.Setenv("OWNER", "acme")
	t.Setenv("REPO", "widgets")
	t.Setenv("SPAM_LABEL_NAME", "junk")
	t.Setenv("MAINTAINERS", "alice, bob ,carol")
	t.Setenv("ISSUE_COUNT", "7")
	t.Setenv("CONCURRENCY_LIMIT", "5")
	t.Setenv("FRESHNESS_WINDOW_DAYS", "30")

	cfg, err := loadConfig([]string{"-dry-run", "-issue", "42"})
	if err != nil {
		t.Fatalf("loadConfig() error = %v", err)
	}
	if cfg.Owner != "acme" || cfg.Repo != "widgets" {
		t.Errorf("owner/repo = %s/%s", cfg.Owner, cfg.Repo)
	}
	if cfg.SpamLabel != "junk" {
		t.Errorf("SpamLabel = %q, want junk", cfg.SpamLabel)
	}
	if !cfg.DryRun {
		t.Error("DryRun = false, want true from flag")
	}
	if cfg.SingleIssue != 42 {
		t.Errorf("SingleIssue = %d, want 42", cfg.SingleIssue)
	}
	if cfg.IssueCount != 7 {
		t.Errorf("IssueCount = %d, want 7", cfg.IssueCount)
	}
	if cfg.Concurrency != 5 {
		t.Errorf("Concurrency = %d, want 5", cfg.Concurrency)
	}
	if cfg.FreshnessWindow != 30*24*time.Hour {
		t.Errorf("FreshnessWindow = %v, want 720h", cfg.FreshnessWindow)
	}
	want := []string{"alice", "bob", "carol"}
	if !reflect.DeepEqual(cfg.Maintainers, want) {
		t.Errorf("Maintainers = %v, want %v", cfg.Maintainers, want)
	}
}

func TestLoadConfigMissingToken(t *testing.T) {
	setRequired(t)
	t.Setenv("GITHUB_TOKEN", "")
	if _, err := loadConfig(nil); err == nil {
		t.Fatal("loadConfig() expected error for missing GITHUB_TOKEN, got nil")
	}
}

func TestLoadConfigVertexFallback(t *testing.T) {
	setRequired(t)
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")
	// No API key, but Vertex enabled -> should succeed.
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
	if _, err := loadConfig(nil); err != nil {
		t.Fatalf("loadConfig() with Vertex fallback error = %v", err)
	}
}

func TestLoadConfigMissingModelCreds(t *testing.T) {
	setRequired(t)
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
	if _, err := loadConfig(nil); err == nil {
		t.Fatal("loadConfig() expected error for missing model credentials, got nil")
	}
}
