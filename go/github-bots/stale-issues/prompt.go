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
	_ "embed"
	"math"
	"strconv"
	"strings"
	"time"
)

//go:embed prompt_instruction.txt
var promptTemplate string

// renderPrompt substitutes the configuration placeholders into the embedded
// prompt and returns a finished instruction string.
//
// The placeholders use {NAME} syntax, which is also how llmagent.Config's
// Instruction field performs session-state templating. We therefore fully
// resolve every placeholder here, leaving no stray braces, so the rendered
// string is safe to pass as a plain Instruction.
func renderPrompt(cfg *Config) string {
	r := strings.NewReplacer(
		"{OWNER}", cfg.Owner,
		"{REPO}", cfg.Repo,
		"{STALE_LABEL_NAME}", cfg.StaleLabel,
		"{REQUEST_CLARIFICATION_LABEL}", cfg.RequestClarificationLabel,
		"{stale_threshold_days}", formatDays(cfg.StaleAfter),
		"{close_threshold_days}", formatDays(cfg.CloseAfter),
	)
	return r.Replace(promptTemplate)
}

// formatDays renders a duration as a clean day count: whole numbers without a
// decimal (e.g. "7"), fractional values with one decimal place (e.g. "0.5").
func formatDays(d time.Duration) string {
	days := d.Hours() / 24
	if days == math.Trunc(days) {
		return strconv.Itoa(int(days))
	}
	return strconv.FormatFloat(days, 'f', 1, 64)
}
