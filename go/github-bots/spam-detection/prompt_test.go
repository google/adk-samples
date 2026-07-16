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
	"strings"
	"testing"
)

// TestRenderPromptNoStrayBraces guards the lesson that
// llmagent.Config.Instruction does {} session-state templating, so a stray
// brace would fail every run.
func TestRenderPromptNoStrayBraces(t *testing.T) {
	cfg := &Config{Owner: "google", Repo: "adk-go", SpamLabel: "spam"}
	out := renderPrompt(cfg)
	if strings.ContainsAny(out, "{}") {
		t.Errorf("rendered prompt contains stray brace(s):\n%s", out)
	}
}

func TestRenderPromptStripsBracesFromConfig(t *testing.T) {
	// A brace arriving via a config value (e.g. an odd spam label) must not leak
	// into the instruction, where llmagent would treat it as a missing state key.
	cfg := &Config{Owner: "google", Repo: "adk-go", SpamLabel: "spam{bot}"}
	if out := renderPrompt(cfg); strings.ContainsAny(out, "{}") {
		t.Errorf("renderPrompt() with a braced label left stray brace(s):\n%s", out)
	}
}

func TestRenderPromptSubstitutesPlaceholders(t *testing.T) {
	cfg := &Config{Owner: "acme", Repo: "widgets", SpamLabel: "junk"}
	out := renderPrompt(cfg)
	for _, want := range []string{"acme/widgets", "'junk'"} {
		if !strings.Contains(out, want) {
			t.Errorf("rendered prompt missing %q", want)
		}
	}
	// Check for the un-substituted {TOKEN} forms, not the bare words: "OWNER"
	// now legitimately appears as a GitHub author-association value.
	if strings.Contains(out, "{OWNER}") || strings.Contains(out, "{SPAM_LABEL_NAME}") {
		t.Errorf("rendered prompt still contains placeholder tokens:\n%s", out)
	}
}
