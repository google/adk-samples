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

// needsTriage reports whether an issue is missing an issue type and/or a
// categorization label from the allowlist. It is pure so it can be exhaustively
// table-tested.
func needsTriage(iss Issue, allowedLabels []string) (needsType, needsLabel bool) {
	needsType = strings.TrimSpace(iss.Type) == ""
	needsLabel = !hasAllowedLabel(iss.Labels, allowedLabels)
	return needsType, needsLabel
}

// hasAllowedLabel reports whether the issue already carries at least one label
// from the allowlist (case-insensitive).
func hasAllowedLabel(labels, allowed []string) bool {
	allowedSet := toLowerSet(allowed)
	for _, l := range labels {
		if _, ok := allowedSet[strings.ToLower(strings.TrimSpace(l))]; ok {
			return true
		}
	}
	return false
}

// isAllowedLabel reports whether label is in the allowlist (case-insensitive).
func isAllowedLabel(label string, allowed []string) bool {
	_, ok := toLowerSet(allowed)[strings.ToLower(strings.TrimSpace(label))]
	return ok
}

// isValidType reports whether t is one of the allowed GitHub issue types
// (case-sensitive; GitHub type names are capitalized, e.g. "Bug").
func isValidType(t string) bool {
	for _, v := range allowedTypes {
		if v == t {
			return true
		}
	}
	return false
}

func toLowerSet(items []string) map[string]struct{} {
	set := make(map[string]struct{}, len(items))
	for _, it := range items {
		set[strings.ToLower(strings.TrimSpace(it))] = struct{}{}
	}
	return set
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
