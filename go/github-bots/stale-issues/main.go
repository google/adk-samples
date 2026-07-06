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
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"golang.org/x/sync/errgroup"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

const (
	appName = "github-stale-bot"
	userID  = "stale-bot"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	if err := run(context.Background(), log, os.Args[1:]); err != nil {
		log.Error("fatal", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, log *slog.Logger, args []string) error {
	// Best-effort: load a local .env when present (for local runs). Ignored in
	// CI, where configuration comes from the environment.
	_ = godotenv.Load()

	cfg, err := loadConfig(args)
	if err != nil {
		return err
	}
	log.Info("starting stale-issue auditor",
		"repo", cfg.Owner+"/"+cfg.Repo, "model", cfg.Model,
		"concurrency", cfg.Concurrency, "dry_run", cfg.DryRun)
	if w := maintainersWarning(cfg); w != "" {
		log.Warn(w)
	}

	gh, err := NewGitHubClient(ctx, cfg, log)
	if err != nil {
		return fmt.Errorf("github client: %w", err)
	}

	tools, err := gh.tools()
	if err != nil {
		return err
	}

	// If a Gemini API key is set it is used directly; otherwise the genai SDK
	// auto-detects its backend (e.g. Vertex AI via ADC) from the environment.
	clientConfig := &genai.ClientConfig{}
	if cfg.GeminiAPIKey != "" {
		clientConfig.APIKey = cfg.GeminiAPIKey
	}
	model, err := gemini.NewModel(ctx, cfg.Model, clientConfig)
	if err != nil {
		return fmt.Errorf("create model: %w", err)
	}

	auditor, err := llmagent.New(llmagent.Config{
		Name:        "stale_issue_auditor",
		Model:       model,
		Description: "Audits open GitHub issues for staleness.",
		Instruction: renderPrompt(cfg),
		Tools:       tools,
		// Temperature 0 keeps the classification deterministic across runs.
		GenerateContentConfig: &genai.GenerateContentConfig{Temperature: genai.Ptr[float32](0)},
	})
	if err != nil {
		return fmt.Errorf("create agent: %w", err)
	}

	sessionService := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          auditor,
		SessionService: sessionService,
	})
	if err != nil {
		return fmt.Errorf("create runner: %w", err)
	}

	issues, err := candidateIssues(ctx, gh, cfg)
	if err != nil {
		return err
	}
	if len(issues) == 0 {
		log.Info("no issues matched the criteria; nothing to do")
		return nil
	}
	log.Info("auditing issues", "count", len(issues))

	auditAll(ctx, r, sessionService, cfg, log, issues)
	return nil
}

// maintainersWarning returns a warning when no maintainers are configured. With
// an empty set, no comment can be classified as a maintainer action, so the bot
// will never mark anything stale (it can still un-stale and alert).
func maintainersWarning(cfg *Config) string {
	if len(cfg.Maintainers) == 0 {
		return "MAINTAINERS is empty: no comment will be treated as maintainer activity, so issues will never be marked stale"
	}
	return ""
}

// candidateIssues returns the issue numbers to audit: either the single issue
// requested via -issue, or all stale candidates from the search.
func candidateIssues(ctx context.Context, gh *GitHubClient, cfg *Config) ([]int, error) {
	if cfg.SingleIssue != 0 {
		return []int{cfg.SingleIssue}, nil
	}
	return gh.SearchOldOpenIssues(ctx)
}

// auditAll audits the issues with bounded concurrency. A failure on one issue is
// logged but never aborts the batch.
func auditAll(ctx context.Context, r *runner.Runner, ss session.Service, cfg *Config, log *slog.Logger, issues []int) {
	g := new(errgroup.Group)
	g.SetLimit(cfg.Concurrency)
	for _, n := range issues {
		g.Go(func() error {
			auditIssue(ctx, r, ss, cfg, log, n)
			return nil
		})
	}
	_ = g.Wait()
	log.Info("audit finished", "processed", len(issues))
}

// auditIssue runs the agent against a single issue in its own fresh session. A
// per-issue session isolates each audit's conversation (its tool calls and the
// model's reasoning) so issues never bleed into each other's context, which also
// lets the bounded-concurrency workers in auditAll run safely in parallel.
func auditIssue(ctx context.Context, r *runner.Runner, ss session.Service, cfg *Config, log *slog.Logger, number int) {
	ictx, cancel := context.WithTimeout(ctx, cfg.IssueTimeout)
	defer cancel()
	// Scope this session to the audited issue so injected instructions in the
	// issue's (untrusted) content cannot make a tool mutate a different issue.
	ictx = withAuditedIssue(ictx, number)
	start := time.Now()
	l := log.With("issue", number)

	resp, err := ss.Create(ictx, &session.CreateRequest{AppName: appName, UserID: userID})
	if err != nil {
		l.Error("create session", "error", err)
		return
	}

	// The issue number reaches the tools *through the model*: this message names
	// the issue, the prompt instructs the model to call get_issue_state, and the
	// model copies the number into each tool's issue_number argument. There is no
	// direct Go call path from here into the tools.
	msg := genai.NewContentFromText(fmt.Sprintf("Audit Issue #%d.", number), genai.RoleUser)

	// r.Run streams every event the agent produces: tool calls, tool results,
	// and model messages. We only want the agent's final natural-language
	// decision, so we keep the text of the last content-bearing event.
	// StreamingModeNone is used because this is a headless batch run with no UI
	// to update token-by-token (cf. StreamingModeSSE in the interactive examples).
	var decision string
	for event, err := range r.Run(ictx, userID, resp.Session.ID(), msg, agent.RunConfig{StreamingMode: agent.StreamingModeNone}) {
		if err != nil {
			l.Error("agent run", "error", err)
			continue
		}
		if event.Content == nil {
			continue
		}
		var b strings.Builder
		for _, p := range event.Content.Parts {
			b.WriteString(p.Text)
		}
		if text := b.String(); text != "" {
			decision = text
		}
	}

	l.Info("audited", "duration", time.Since(start).Round(time.Millisecond), "decision", summarize(decision))
}

// summarize collapses the agent's final text into a single short log line.
func summarize(s string) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	const max = 200
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}
