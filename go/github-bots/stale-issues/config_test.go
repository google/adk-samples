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

// setRequiredCreds sets the mandatory credentials and clears optional vars so
// tests observe defaults rather than the ambient environment.
func setRequiredCreds(t *testing.T) {
	t.Helper()
	t.Setenv("GITHUB_TOKEN", "tok")
	t.Setenv("GEMINI_API_KEY", "key")
	// OWNER/REPO are required (no default), so they are part of the minimum.
	t.Setenv("OWNER", "google")
	t.Setenv("REPO", "adk-go")
	for _, k := range []string{
		"GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "LLM_MODEL_NAME",
		"STALE_HOURS_THRESHOLD", "CLOSE_HOURS_AFTER_STALE_THRESHOLD",
		"STALE_LABEL_NAME", "REQUEST_CLARIFICATION_LABEL", "MAINTAINERS",
		"CONCURRENCY_LIMIT", "ISSUE_TIMEOUT", "DRY_RUN",
	} {
		t.Setenv(k, "")
	}
}

func TestLoadConfig_Defaults(t *testing.T) {
	setRequiredCreds(t)
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.Model != "gemini-flash-latest" {
		t.Errorf("Model = %q, want gemini-flash-latest", cfg.Model)
	}
	if cfg.Owner != "google" || cfg.Repo != "adk-go" {
		t.Errorf("Owner/Repo = %q/%q, want google/adk-go", cfg.Owner, cfg.Repo)
	}
	if cfg.StaleAfter != 14*24*time.Hour {
		t.Errorf("StaleAfter = %v, want 336h (14d)", cfg.StaleAfter)
	}
	if cfg.CloseAfter != 7*24*time.Hour {
		t.Errorf("CloseAfter = %v, want 168h (7d)", cfg.CloseAfter)
	}
	if cfg.Concurrency != 3 {
		t.Errorf("Concurrency = %d, want 3", cfg.Concurrency)
	}
	if cfg.DryRun {
		t.Error("DryRun = true, want false by default")
	}
}

func TestLoadConfig_ParsesValues(t *testing.T) {
	setRequiredCreds(t)
	t.Setenv("STALE_HOURS_THRESHOLD", "48")
	t.Setenv("MAINTAINERS", "alice, bob ,carol")
	t.Setenv("LLM_MODEL_NAME", "custom-model")

	cfg, err := loadConfig([]string{"-dry-run", "-issue", "123"})
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.StaleAfter != 48*time.Hour {
		t.Errorf("StaleAfter = %v, want 48h", cfg.StaleAfter)
	}
	if want := []string{"alice", "bob", "carol"}; !reflect.DeepEqual(cfg.Maintainers, want) {
		t.Errorf("Maintainers = %v, want %v (trimmed)", cfg.Maintainers, want)
	}
	if cfg.Model != "custom-model" {
		t.Errorf("Model = %q, want custom-model", cfg.Model)
	}
	if !cfg.DryRun {
		t.Error("DryRun = false, want true (flag)")
	}
	if cfg.SingleIssue != 123 {
		t.Errorf("SingleIssue = %d, want 123", cfg.SingleIssue)
	}
}

func TestLoadConfig_MissingCredentials(t *testing.T) {
	t.Setenv("GITHUB_TOKEN", "")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")
	if _, err := loadConfig(nil); err == nil {
		t.Fatal("expected error when GITHUB_TOKEN and GEMINI_API_KEY are missing")
	}
}

func TestLoadConfig_MissingOwnerRepo(t *testing.T) {
	for _, unset := range []string{"OWNER", "REPO"} {
		t.Run(unset, func(t *testing.T) {
			setRequiredCreds(t)
			t.Setenv(unset, "")
			if _, err := loadConfig(nil); err == nil {
				t.Fatalf("loadConfig() with empty %s = nil error, want error (no silent default)", unset)
			}
		})
	}
}
