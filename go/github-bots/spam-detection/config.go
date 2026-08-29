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

// Config holds all runtime configuration for the spam-detection bot. It is
// parsed once from environment variables and command-line flags and then
// injected into the rest of the program; there is deliberately no
// package-level mutable state.
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

	// SpamLabel is the label applied to issues judged to be spam. It must
	// already exist in the repository.
	SpamLabel string

	// Maintainers is the set of GitHub logins whose comments are trusted and
	// therefore never reviewed for spam. The default GITHUB_TOKEN cannot list
	// collaborators, so the set is supplied explicitly via the MAINTAINERS env
	// var (comma-separated). When empty, maintainer comments are reviewed like
	// anyone else's (see maintainersWarning).
	Maintainers []string

	// IssueCount caps how many candidate issues a single scheduled sweep
	// processes (most-recently-updated first).
	IssueCount int

	// FreshnessWindow optionally restricts the sweep to issues updated within
	// the window. Zero disables the restriction (the full open backlog). Spam
	// frequently arrives as a comment on an older issue, so the window filters
	// on last-updated time rather than creation time.
	FreshnessWindow time.Duration

	// Concurrency bounds how many issues are reviewed in parallel.
	Concurrency int

	// IssueTimeout bounds how long a single issue review may take.
	IssueTimeout time.Duration

	// DryRun, when true, logs intended mutations without performing them.
	DryRun bool

	// SingleIssue, when non-zero, reviews only that issue and skips the search
	// step. Useful for local testing and workflow_dispatch.
	SingleIssue int
}

// loadConfig parses configuration from flags (args) and environment variables
// and validates required fields. args is injectable so tests can exercise flag
// parsing.
// defaultIssueTimeout bounds one issue's agent run. It is both the default and
// the value a non-positive ISSUE_TIMEOUT is clamped back to, so the two must not
// drift apart.
const defaultIssueTimeout = 5 * time.Minute

func loadConfig(args []string) (*Config, error) {
	fs := flag.NewFlagSet("githubspambot", flag.ContinueOnError)
	dryRun := fs.Bool("dry-run", envBool("DRY_RUN", false),
		"log intended actions without labeling or commenting")
	singleIssue := fs.Int("issue", 0,
		"review only this issue number and skip the search step (0 = sweep candidates)")
	if err := fs.Parse(args); err != nil {
		return nil, err
	}

	cfg := &Config{
		// OWNER/REPO have no default on purpose: a default would silently target a
		// concrete repository if a caller forgot to set them (validate() rejects an
		// empty value instead, so misconfiguration fails loudly).
		Owner:           os.Getenv("OWNER"),
		Repo:            os.Getenv("REPO"),
		GitHubToken:     os.Getenv("GITHUB_TOKEN"),
		GeminiAPIKey:    firstNonEmpty(os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY")),
		Model:           getenv("LLM_MODEL_NAME", "gemini-flash-latest"),
		SpamLabel:       getenv("SPAM_LABEL_NAME", "spam"),
		Maintainers:     splitList(os.Getenv("MAINTAINERS")),
		IssueCount:      envInt("ISSUE_COUNT", 3),
		FreshnessWindow: envDays("FRESHNESS_WINDOW_DAYS", 0),
		Concurrency:     envInt("CONCURRENCY_LIMIT", 3),
		IssueTimeout:    envDuration("ISSUE_TIMEOUT", defaultIssueTimeout),
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
	if c.SpamLabel == "" {
		missing = append(missing, "SPAM_LABEL_NAME")
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required configuration: %s", strings.Join(missing, ", "))
	}
	if c.IssueCount < 1 {
		c.IssueCount = 1
	}
	if c.Concurrency < 1 {
		c.Concurrency = 1
	}
	if c.FreshnessWindow < 0 {
		c.FreshnessWindow = 0
	}
	if c.IssueTimeout <= 0 {
		c.IssueTimeout = defaultIssueTimeout
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

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
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

func envDuration(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

// envDays reads a (possibly fractional) number of days and returns a Duration.
func envDays(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if days, err := strconv.ParseFloat(v, 64); err == nil {
			return time.Duration(days * float64(24*time.Hour))
		}
	}
	return def
}
