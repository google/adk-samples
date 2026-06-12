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
			gotType, gotLabel := needsTriage(tc.issue, allowed)
			if gotType != tc.wantNeedsType || gotLabel != tc.wantNeedsLabel {
				t.Errorf("needsTriage() = (type:%t, label:%t), want (type:%t, label:%t)",
					gotType, gotLabel, tc.wantNeedsType, tc.wantNeedsLabel)
			}
		})
	}
}

func TestIsAllowedLabel(t *testing.T) {
	allowed := []string{"bug", "enhancement"}
	tests := []struct {
		label string
		want  bool
	}{
		{"bug", true},
		{"BUG", true},
		{" enhancement ", true},
		{"documentation", false},
		{"", false},
	}
	for _, tc := range tests {
		if got := isAllowedLabel(tc.label, allowed); got != tc.want {
			t.Errorf("isAllowedLabel(%q) = %t, want %t", tc.label, got, tc.want)
		}
	}
}

func TestIsValidType(t *testing.T) {
	tests := []struct {
		t    string
		want bool
	}{
		{"Bug", true},
		{"Feature", true},
		{"Task", true},
		{"bug", false}, // case-sensitive: GitHub type names are capitalized
		{"Epic", false},
		{"", false},
	}
	for _, tc := range tests {
		if got := isValidType(tc.t); got != tc.want {
			t.Errorf("isValidType(%q) = %t, want %t", tc.t, got, tc.want)
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
