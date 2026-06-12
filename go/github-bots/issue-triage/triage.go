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

import "strings"

// Issue is the normalized view of a GitHub issue used for triage decisions and
// returned to the model. It is deliberately small: only the fields needed to
// classify and act. The json tags shape what the model sees.
type Issue struct {
	Number int      `json:"number"`
	Title  string   `json:"title"`
	Body   string   `json:"body"`
	Labels []string `json:"labels"`
	// Type is the GitHub issue type name (e.g. "Bug"), or "" when unset.
	Type string `json:"type"`
}

// need records which fields an issue still requires. It lets the agent fill
// only the gaps: authorization carries the need so the tools can reject an
// attempt to overwrite a type or label that is already set (enforced in code,
// not merely requested in the prompt).
type need struct {
	typ   bool
	label bool
}

func (n need) any() bool { return n.typ || n.label }

// needsTriage reports which fields an issue is missing: an issue type and/or a
// categorization label from the allowlist. It is pure so it can be exhaustively
// table-tested.
func needsTriage(iss Issue, allowedLabels []string) need {
	return need{
		typ:   strings.TrimSpace(iss.Type) == "",
		label: !hasAllowedLabel(iss.Labels, allowedLabels),
	}
}

// hasAllowedLabel reports whether the issue already carries at least one label
// from the allowlist (case-insensitive).
func hasAllowedLabel(labels, allowed []string) bool {
	for _, l := range labels {
		if _, ok := canonicalLabel(l, allowed); ok {
			return true
		}
	}
	return false
}

// canonicalLabel matches label against the allowlist case-insensitively and
// returns the allowlist's spelling, so GitHub always receives the label name
// exactly as it exists in the repository (regardless of the model's casing).
func canonicalLabel(label string, allowed []string) (string, bool) {
	label = strings.TrimSpace(label)
	for _, a := range allowed {
		if strings.EqualFold(a, label) {
			return a, true
		}
	}
	return "", false
}

// canonicalType matches t against the allowed GitHub issue types
// case-insensitively and returns the canonical name (GitHub type names are
// capitalized, e.g. "Bug"). Accepting any casing makes the bot robust to model
// output variance while always sending GitHub the exact name.
func canonicalType(t string) (string, bool) {
	t = strings.TrimSpace(t)
	for _, v := range allowedTypes {
		if strings.EqualFold(v, t) {
			return v, true
		}
	}
	return "", false
}

// truncate shortens s to at most n runes, appending an ellipsis marker when it
// trims. Keeps very long issue bodies from bloating the prompt.
func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "\n…[truncated]"
}
