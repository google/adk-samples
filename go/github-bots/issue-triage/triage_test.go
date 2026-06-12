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

import "testing"

func TestNeedsTriage(t *testing.T) {
	allowed := []string{"bug", "enhancement", "documentation", "question"}
	tests := []struct {
		name           string
		issue          Issue
		wantNeedsType  bool
		wantNeedsLabel bool
	}{
		{
			name:           "no type and no label",
			issue:          Issue{Number: 1},
			wantNeedsType:  true,
			wantNeedsLabel: true,
		},
		{
			name:           "fully triaged",
			issue:          Issue{Number: 2, Type: "Bug", Labels: []string{"bug"}},
			wantNeedsType:  false,
			wantNeedsLabel: false,
		},
		{
			name:           "has type, missing label",
			issue:          Issue{Number: 3, Type: "Feature", Labels: []string{"go"}},
			wantNeedsType:  false,
			wantNeedsLabel: true,
		},
		{
			name:           "has label, missing type",
			issue:          Issue{Number: 4, Labels: []string{"enhancement"}},
			wantNeedsType:  true,
			wantNeedsLabel: false,
		},
		{
			name:           "label match is case-insensitive",
			issue:          Issue{Number: 5, Type: "Bug", Labels: []string{"BUG"}},
			wantNeedsType:  false,
			wantNeedsLabel: false,
		},
		{
			name:           "whitespace-only type counts as missing",
			issue:          Issue{Number: 6, Type: "  ", Labels: []string{"bug"}},
			wantNeedsType:  true,
			wantNeedsLabel: false,
		},
		{
			name:           "non-allowlisted label does not count",
			issue:          Issue{Number: 7, Type: "Bug", Labels: []string{"good first issue"}},
			wantNeedsType:  false,
			wantNeedsLabel: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := needsTriage(tc.issue, allowed)
			if got.typ != tc.wantNeedsType || got.label != tc.wantNeedsLabel {
				t.Errorf("needsTriage() = (type:%t, label:%t), want (type:%t, label:%t)",
					got.typ, got.label, tc.wantNeedsType, tc.wantNeedsLabel)
			}
		})
	}
}

func TestCanonicalLabel(t *testing.T) {
	allowed := []string{"bug", "enhancement"}
	tests := []struct {
		label    string
		wantOK   bool
		wantName string
	}{
		{"bug", true, "bug"},
		{"BUG", true, "bug"}, // canonicalized to the allowlist's spelling
		{" enhancement ", true, "enhancement"},
		{"documentation", false, ""},
		{"", false, ""},
	}
	for _, tc := range tests {
		got, ok := canonicalLabel(tc.label, allowed)
		if ok != tc.wantOK || got != tc.wantName {
			t.Errorf("canonicalLabel(%q) = (%q, %t), want (%q, %t)", tc.label, got, ok, tc.wantName, tc.wantOK)
		}
	}
}

func TestCanonicalType(t *testing.T) {
	tests := []struct {
		t        string
		wantOK   bool
		wantName string
	}{
		{"Bug", true, "Bug"},
		{"Feature", true, "Feature"},
		{"Task", true, "Task"},
		{"bug", true, "Bug"}, // any casing is accepted and canonicalized
		{" task ", true, "Task"},
		{"Epic", false, ""},
		{"", false, ""},
	}
	for _, tc := range tests {
		got, ok := canonicalType(tc.t)
		if ok != tc.wantOK || got != tc.wantName {
			t.Errorf("canonicalType(%q) = (%q, %t), want (%q, %t)", tc.t, got, ok, tc.wantName, tc.wantOK)
		}
	}
}

func TestTruncate(t *testing.T) {
	if got := truncate("hello", 10); got != "hello" {
		t.Errorf("truncate short string changed it: %q", got)
	}
	got := truncate("hello world", 5)
	if got == "hello world" {
		t.Errorf("truncate did not shorten: %q", got)
	}
	if len([]rune(got)) <= 5 {
		t.Errorf("truncate produced no marker: %q", got)
	}
	// Multi-byte runes must not be split mid-character.
	if got := truncate("héllo wörld", 4); []rune(got)[3] != 'l' {
		t.Errorf("truncate split a rune boundary: %q", got)
	}
}
