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

// renderPrompt substitutes the configuration placeholders into the embedded
// prompt and returns a finished instruction string.
//
// IMPORTANT: llmagent.Config.Instruction treats {placeholder} tokens as
// session-state references and errors on unknown keys. renderPrompt must
// therefore leave zero stray braces; this is enforced by a test.
func renderPrompt(cfg *Config) string {
	// Strip braces from config-derived values: llmagent treats { and } as
	// session-state references, so a brace arriving via SPAM_LABEL_NAME/OWNER/REPO
	// (e.g. a label like "spam{bot}") would otherwise inject an unknown state key
	// and fail every run.
	r := strings.NewReplacer(
		"{OWNER}", stripBraces(cfg.Owner),
		"{REPO}", stripBraces(cfg.Repo),
		"{SPAM_LABEL_NAME}", stripBraces(cfg.SpamLabel),
	)
	return r.Replace(promptTemplate)
}

var braceStripper = strings.NewReplacer("{", "", "}", "")

// stripBraces removes brace characters so a substituted value cannot be parsed
// as an llmagent session-state placeholder.
func stripBraces(s string) string { return braceStripper.Replace(s) }
