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
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// defaultAllowedLabels are the categorization labels the bot may apply. They
// must already exist in the target repository.
var defaultAllowedLabels = []string{"bug", "enhancement", "documentation", "question"}

// allowedTypes are the GitHub issue types the bot may set. These must be
// enabled at the organization level.
var allowedTypes = []string{"Bug", "Feature", "Task"}

// Config holds all runtime configuration. It is parsed once and injected; there
// is no package-level mutable state.
type Config struct {
	Owner string
	Repo  string

	GitHubToken  string
	GeminiAPIKey string
	Model        string

	// AllowedLabels is the set of categorization labels the agent may apply.
	AllowedLabels []string

	// IssueCount caps how many untriaged issues a single scheduled sweep
	// processes (newest first).
	IssueCount int
	// FreshnessWindow optionally restricts the sweep to issues created within
	// the window. Zero disables the restriction (full backlog).
	FreshnessWindow time.Duration
	// IssueTimeout bounds a single agent run.
	IssueTimeout time.Duration

	// DryRun logs intended actions without performing any mutation.
	DryRun bool
	// SingleIssue, when > 0, triages only that issue instead of sweeping.
	SingleIssue int
}

// loadConfig parses configuration from flags (args) and environment variables.
// args is injectable so tests can exercise flag parsing.
func loadConfig(args []string) (*Config, error) {
	fs := flag.NewFlagSet("githubtriagebot", flag.ContinueOnError)
	dryRun := fs.Bool("dry-run", envBool("DRY_RUN", false), "Log intended actions without modifying any issues.")
	singleIssue := fs.Int("issue", 0, "Triage only this issue number (0 = sweep untriaged issues).")
	if err := fs.Parse(args); err != nil {
		return nil, err
	}

	cfg := &Config{
		Owner:           getenv("OWNER", "google"),
		Repo:            getenv("REPO", "adk-go"),
		GitHubToken:     os.Getenv("GITHUB_TOKEN"),
		GeminiAPIKey:    firstNonEmpty(os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY")),
		Model:           getenv("LLM_MODEL_NAME", "gemini-3.5-flash"),
		AllowedLabels:   splitList(getenv("ALLOWED_LABELS", strings.Join(defaultAllowedLabels, ","))),
		IssueCount:      envInt("ISSUE_COUNT", 3),
		FreshnessWindow: envDays("FRESHNESS_WINDOW_DAYS", 0),
		IssueTimeout:    envDuration("ISSUE_TIMEOUT", 5*time.Minute),
		DryRun:          *dryRun,
		SingleIssue:     *singleIssue,
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

func (c *Config) validate() error {
	var missing []string
	if c.GitHubToken == "" {
		missing = append(missing, "GITHUB_TOKEN")
	}
	if c.Owner == "" {
		missing = append(missing, "OWNER")
	}
	if c.Repo == "" {
		missing = append(missing, "REPO")
	}
	// A Gemini API key is the simplest path, but Vertex AI via ADC is also
	// supported; in that case the genai SDK reads its configuration from the
	// environment (GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, ...).
	if c.GeminiAPIKey == "" && !envBool("GOOGLE_GENAI_USE_VERTEXAI", false) {
		missing = append(missing, "GEMINI_API_KEY (or set GOOGLE_GENAI_USE_VERTEXAI=true for Vertex AI)")
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required configuration: %s", strings.Join(missing, ", "))
	}

	if c.IssueCount < 1 {
		c.IssueCount = 1
	}
	if len(c.AllowedLabels) == 0 {
		c.AllowedLabels = defaultAllowedLabels
	}
	return nil
}

// Environment helpers.

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// splitList splits a comma-separated list, trimming whitespace and dropping
// empty entries.
func splitList(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func envInt(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

func envBool(key string, def bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return def
	}
	return b
}

func envDuration(key string, def time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return def
	}
	return d
}

// envDays reads a (possibly fractional) number of days and returns a Duration.
func envDays(key string, def time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	days, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return def
	}
	return time.Duration(days * float64(24*time.Hour))
}
