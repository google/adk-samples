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
	"strings"
)

//go:embed prompt_instruction.txt
var promptTemplate string

// renderPrompt substitutes every placeholder in the embedded instruction.
//
// IMPORTANT: llmagent.Config.Instruction treats {placeholder} tokens as
// session-state references and errors on unknown keys. renderPrompt must
// therefore leave zero stray braces; this is enforced by a test.
//
// Note: prompt_instruction.txt only parameterizes the label/type *names*. Its
// per-label rubric ("How to choose the categorization label") is written for the
// default labels, so overriding ALLOWED_LABELS with a different vocabulary also
// warrants updating that section.
func renderPrompt(cfg *Config) string {
	// Strip braces from config-derived values: llmagent treats { and } as
	// session-state references, so a brace arriving via OWNER/REPO/ALLOWED_LABELS
	// would otherwise reintroduce the stray-brace failure the test above guards.
	r := strings.NewReplacer(
		"{OWNER}", stripBraces(cfg.Owner),
		"{REPO}", stripBraces(cfg.Repo),
		"{ALLOWED_LABELS}", stripBraces(strings.Join(cfg.AllowedLabels, ", ")),
		"{ALLOWED_TYPES}", strings.Join(allowedTypes, ", "),
	)
	return r.Replace(promptTemplate)
}

var braceStripper = strings.NewReplacer("{", "", "}", "")

// stripBraces removes brace characters so a substituted value cannot be parsed
// as an llmagent session-state placeholder.
func stripBraces(s string) string { return braceStripper.Replace(s) }
