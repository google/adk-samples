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
	"testing"
)

func TestAuthorizeIssue(t *testing.T) {
	// No issue bound to the context (e.g. a tool call outside an audit session).
	if _, ok := authorizeIssue(context.Background(), 7); ok {
		t.Error("authorizeIssue with no bound issue = ok, want rejected")
	}

	ctx := withAuditedIssue(context.Background(), 7)
	if _, ok := authorizeIssue(ctx, 7); !ok {
		t.Error("authorizeIssue(7) on a session scoped to 7 = rejected, want ok")
	}
	// The defense against prompt injection: a different issue must be refused.
	if msg, ok := authorizeIssue(ctx, 8); ok {
		t.Errorf("authorizeIssue(8) on a session scoped to 7 = ok, want rejected (msg=%q)", msg)
	}
}

func TestIsManagedLabel(t *testing.T) {
	c := &GitHubClient{cfg: &Config{
		StaleLabel:                "stale",
		RequestClarificationLabel: "request clarification",
	}}
	tests := []struct {
		label string
		want  bool
	}{
		{"stale", true},
		{"request clarification", true},
		{"security", false},
		{"release-blocker", false},
		{"", false},
	}
	for _, tc := range tests {
		if got := c.isManagedLabel(tc.label); got != tc.want {
			t.Errorf("isManagedLabel(%q) = %t, want %t", tc.label, got, tc.want)
		}
	}
}
