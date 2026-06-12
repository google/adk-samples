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
	"time"
)

func promptCfg() *Config {
	return &Config{
		Owner:                     "google",
		Repo:                      "adk-go",
		StaleLabel:                "stale",
		RequestClarificationLabel: "request clarification",
		StaleAfter:                168 * time.Hour,
		CloseAfter:                168 * time.Hour,
	}
}

// The rendered prompt is passed to llmagent as Instruction, which performs {}
// session-state templating. Any leftover brace would be treated as a missing
// state key and fail every run, so the render must leave none. This guards
// against adding a new {placeholder} to the prompt without a replacer entry.
func TestRenderPrompt_NoStrayBraces(t *testing.T) {
	out := renderPrompt(promptCfg())
	if i := strings.IndexAny(out, "{}"); i != -1 {
		start := i - 30
		if start < 0 {
			start = 0
		}
		end := i + 30
		if end > len(out) {
			end = len(out)
		}
		t.Errorf("rendered prompt still contains a brace near: %q", out[start:end])
	}
}

func TestRenderPrompt_SubstitutesPlaceholders(t *testing.T) {
	out := renderPrompt(promptCfg())
	for _, want := range []string{"google/adk-go", "'stale'", "'request clarification'", "7 days"} {
		if !strings.Contains(out, want) {
			t.Errorf("rendered prompt missing %q", want)
		}
	}
}

func TestFormatDays(t *testing.T) {
	cases := map[time.Duration]string{
		168 * time.Hour: "7",
		24 * time.Hour:  "1",
		12 * time.Hour:  "0.5",
		36 * time.Hour:  "1.5",
	}
	for d, want := range cases {
		if got := formatDays(d); got != want {
			t.Errorf("formatDays(%v) = %q, want %q", d, got, want)
		}
	}
}
