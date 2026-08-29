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

// Config holds all runtime configuration for the stale-issue bot. It is parsed
// once from environment variables and command-line flags and then injected into
// the rest of the program; there is deliberately no package-level mutable state.
type Config struct {
	// Owner and Repo identify the target repository (e.g. "google"/"adk-go").
	Owner string
	Repo  string

	// GitHubToken authenticates GitHub REST and GraphQL calls. In GitHub
	// Actions this is the auto-provided github-actions[bot] token
	// (${{ secrets.GITHUB_TOKEN }}), authorized via the workflow permissions
	// block.
	GitHubToken string

	// GeminiAPIKey authenticates the Gemini (AI Studio) model.
	GeminiAPIKey string

	// Model is the Gemini model name used for reasoning.
	Model string

	// StaleAfter is how long an issue may sit waiting on the author (after a
	// maintainer's request) before it is marked stale. Default: 14 days.
	StaleAfter time.Duration

	// CloseAfter is how long an issue may remain stale (after the warning
	// comment) before it is closed. Default: 7 days.
	CloseAfter time.Duration

	// StaleLabel and RequestClarificationLabel are the label names the bot
	// manages. They must already exist in the repository.
	StaleLabel                string
	RequestClarificationLabel string

	// Maintainers is the set of GitHub logins treated as maintainers. The
	// default GITHUB_TOKEN cannot list collaborators, so the maintainer set is
	// supplied explicitly via the MAINTAINERS env var (comma-separated).
	Maintainers []string

	// Concurrency bounds how many issues are audited in parallel.
	Concurrency int

	// IssueTimeout bounds how long a single issue audit may take.
	IssueTimeout time.Duration

	// DryRun, when true, logs intended mutations without performing them.
	DryRun bool

	// SingleIssue, when non-zero, audits only that issue and skips the search
	// step. Useful for local testing and workflow_dispatch.
	SingleIssue int
}

// loadConfig parses configuration from flags and environment variables and
// validates required fields.
func loadConfig(args []string) (*Config, error) {
	fs := flag.NewFlagSet("githubstalebot", flag.ContinueOnError)
	dryRun := fs.Bool("dry-run", envBool("DRY_RUN", false),
		"log intended actions without commenting, labeling, or closing")
	singleIssue := fs.Int("issue", 0,
		"audit only this issue number and skip the search step (0 = audit all stale candidates)")
	if err := fs.Parse(args); err != nil {
		return nil, err
	}

	cfg := &Config{
		// OWNER/REPO have no default on purpose: a default would silently target a
		// concrete repository if a caller forgot to set them (validate() rejects an
		// empty value instead, so misconfiguration fails loudly).
		Owner:                     os.Getenv("OWNER"),
		Repo:                      os.Getenv("REPO"),
		GitHubToken:               os.Getenv("GITHUB_TOKEN"),
		GeminiAPIKey:              firstNonEmpty(os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY")),
		Model:                     getenv("LLM_MODEL_NAME", "gemini-flash-latest"),
		StaleAfter:                envHours("STALE_HOURS_THRESHOLD", 14*24*time.Hour),
		CloseAfter:                envHours("CLOSE_HOURS_AFTER_STALE_THRESHOLD", 7*24*time.Hour),
		StaleLabel:                getenv("STALE_LABEL_NAME", "stale"),
		RequestClarificationLabel: getenv("REQUEST_CLARIFICATION_LABEL", "request clarification"),
		Maintainers:               splitList(os.Getenv("MAINTAINERS")),
		Concurrency:               envInt("CONCURRENCY_LIMIT", 3),
		IssueTimeout:              envDuration("ISSUE_TIMEOUT", 5*time.Minute),
		DryRun:                    *dryRun,
		SingleIssue:               *singleIssue,
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
	// A Gemini API key is the simplest path, but Vertex AI via ADC is also
	// supported; in that case the genai SDK reads its configuration from the
	// environment (GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, ...).
	if c.GeminiAPIKey == "" && !envBool("GOOGLE_GENAI_USE_VERTEXAI", false) {
		missing = append(missing, "GEMINI_API_KEY (or set GOOGLE_GENAI_USE_VERTEXAI=true for Vertex AI)")
	}
	if c.Owner == "" {
		missing = append(missing, "OWNER")
	}
	if c.Repo == "" {
		missing = append(missing, "REPO")
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required configuration: %s", strings.Join(missing, ", "))
	}
	// Reject non-positive durations rather than coercing them. A negative
	// STALE_HOURS_THRESHOLD — a plausible sign typo, plumbed straight from a
	// workflow input — puts the search cutoff in the future, so `created:<cutoff`
	// matches every open issue including ones opened minutes ago, and the
	// rendered threshold makes every comparison true. One character would turn
	// the bot into a mass-marker, so this has to fail loudly.
	if c.StaleAfter <= 0 {
		return fmt.Errorf("STALE_HOURS_THRESHOLD must be positive, got %s", c.StaleAfter)
	}
	if c.CloseAfter <= 0 {
		return fmt.Errorf("CLOSE_HOURS_AFTER_STALE_THRESHOLD must be positive, got %s", c.CloseAfter)
	}
	if c.IssueTimeout <= 0 {
		return fmt.Errorf("ISSUE_TIMEOUT must be positive, got %s", c.IssueTimeout)
	}
	if c.StaleLabel != "" && strings.EqualFold(c.StaleLabel, c.RequestClarificationLabel) {
		return fmt.Errorf("STALE_LABEL_NAME (%q) and REQUEST_CLARIFICATION_LABEL (%q) must differ; GitHub label names are case-insensitively unique, so a collision would route clarification-label writes through the staleness gate", c.StaleLabel, c.RequestClarificationLabel)
	}
	if c.Concurrency < 1 {
		c.Concurrency = 1
	}
	return nil
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

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
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}

// envHours reads a float number of hours from the environment and returns it as
// a time.Duration. Thresholds are expressed in hours for easy configuration in
// the workflow (e.g. STALE_HOURS_THRESHOLD=168).
func envHours(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if hours, err := strconv.ParseFloat(v, 64); err == nil {
			return time.Duration(hours * float64(time.Hour))
		}
	}
	return def
}

func envDuration(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
