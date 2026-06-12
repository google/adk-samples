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
	"testing"
	"time"
)

var (
	testNow        = time.Date(2026, 1, 31, 12, 0, 0, 0, time.UTC)
	testStaleAfter = 168 * time.Hour // 7 days
	testCloseAfter = 168 * time.Hour // 7 days
	testStaleLabel = "stale"
	testSelf       = "stale-bot[bot]"
	testAuthor     = "reporter"
	testMaint      = []string{"maintainerA", "maintainerB"}
)

func actor(login string) *rawActor { return &rawActor{Login: login} }

// daysAgo returns a timestamp the given number of days before testNow.
func daysAgo(d float64) time.Time {
	return testNow.Add(-time.Duration(d * float64(24*time.Hour)))
}

func TestComputeIssueState(t *testing.T) {
	tests := []struct {
		name            string
		raw             *rawIssue
		wantRole        Role
		wantAction      eventType
		wantStale       bool
		wantAlert       bool
		wantComment     string
		minDaysActivity float64 // assert DaysSinceActivity >= this (0 to skip)
	}{
		{
			name: "author commented last is active",
			raw: &rawIssue{
				Author:    actor(testAuthor),
				CreatedAt: daysAgo(30),
				Comments: commentNodes(
					rawComment{Author: actor("maintainerA"), Body: "Can you provide logs?", CreatedAt: daysAgo(20)},
					rawComment{Author: actor(testAuthor), Body: "Here are the logs", CreatedAt: daysAgo(1)},
				),
			},
			wantRole:    roleAuthor,
			wantAction:  eventCommented,
			wantStale:   false,
			wantComment: "Here are the logs",
		},
		{
			name: "maintainer question past threshold",
			raw: &rawIssue{
				Author:    actor(testAuthor),
				CreatedAt: daysAgo(30),
				Comments: commentNodes(
					rawComment{Author: actor("maintainerA"), Body: "Please share a repro.", CreatedAt: daysAgo(10)},
				),
			},
			wantRole:        roleMaintainer,
			wantAction:      eventCommented,
			wantStale:       false,
			wantComment:     "Please share a repro.",
			minDaysActivity: 9,
		},
		{
			name: "silent description edit needs alert",
			raw: &rawIssue{
				Author:    actor(testAuthor),
				CreatedAt: daysAgo(30),
				UserContentEdits: editNodes(
					rawEdit{Editor: actor(testAuthor), EditedAt: daysAgo(2)},
				),
			},
			wantRole:   roleAuthor,
			wantAction: eventEditedDesc,
			wantAlert:  true,
		},
		{
			name: "silent edit already alerted does not re-alert",
			raw: &rawIssue{
				Author:    actor(testAuthor),
				CreatedAt: daysAgo(30),
				UserContentEdits: editNodes(
					rawEdit{Editor: actor(testAuthor), EditedAt: daysAgo(3)},
				),
				Comments: commentNodes(
					rawComment{Author: actor(testSelf), Body: botAlertSignature + ". Maintainers, please review.", CreatedAt: daysAgo(2)},
				),
			},
			wantRole:   roleAuthor,
			wantAction: eventEditedDesc,
			wantAlert:  false,
		},
		{
			name: "bot and self comments are ignored",
			raw: &rawIssue{
				Author:    actor(testAuthor),
				CreatedAt: daysAgo(30),
				Comments: commentNodes(
					rawComment{Author: actor("maintainerA"), Body: "Any update?", CreatedAt: daysAgo(10)},
					rawComment{Author: actor("github-actions[bot]"), Body: "automated note", CreatedAt: daysAgo(1)},
					rawComment{Author: actor(testSelf), Body: "another automated note", CreatedAt: daysAgo(1)},
				),
			},
			wantRole:    roleMaintainer,
			wantAction:  eventCommented,
			wantComment: "Any update?",
		},
		{
			name: "last comment text retained across later non-comment event",
			raw: &rawIssue{
				Author:    actor(testAuthor),
				CreatedAt: daysAgo(30),
				Comments: commentNodes(
					rawComment{Author: actor("maintainerA"), Body: "Could you clarify the use case?", CreatedAt: daysAgo(10)},
				),
				TimelineItems: timelineNodes(
					rawTimelineItem{Typename: "RenamedTitleEvent", Actor: actor("maintainerA"), CreatedAt: daysAgo(9)},
				),
			},
			wantRole:    roleMaintainer,
			wantAction:  eventRenamedTitle,
			wantComment: "Could you clarify the use case?", // retained despite rename being last
		},
		{
			name: "stale label present",
			raw: &rawIssue{
				Author:    actor(testAuthor),
				CreatedAt: daysAgo(30),
				Labels:    labelNodes(testStaleLabel),
				Comments: commentNodes(
					rawComment{Author: actor("maintainerA"), Body: "Marking stale.", CreatedAt: daysAgo(8)},
				),
				TimelineItems: timelineNodes(
					rawTimelineItem{Typename: "LabeledEvent", CreatedAt: daysAgo(8), Label: &struct {
						Name string `json:"name"`
					}{Name: testStaleLabel}},
				),
			},
			wantRole:   roleMaintainer,
			wantAction: eventCommented,
			wantStale:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := computeIssueState(tt.raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)

			if got.LastActionRole != string(tt.wantRole) {
				t.Errorf("LastActionRole = %q, want %q", got.LastActionRole, tt.wantRole)
			}
			if got.LastActionType != string(tt.wantAction) {
				t.Errorf("LastActionType = %q, want %q", got.LastActionType, tt.wantAction)
			}
			if got.IsStale != tt.wantStale {
				t.Errorf("IsStale = %v, want %v", got.IsStale, tt.wantStale)
			}
			if got.MaintainerAlertNeeded != tt.wantAlert {
				t.Errorf("MaintainerAlertNeeded = %v, want %v", got.MaintainerAlertNeeded, tt.wantAlert)
			}
			if tt.wantComment != "" && got.LastCommentText != tt.wantComment {
				t.Errorf("LastCommentText = %q, want %q", got.LastCommentText, tt.wantComment)
			}
			if tt.minDaysActivity > 0 && got.DaysSinceActivity < tt.minDaysActivity {
				t.Errorf("DaysSinceActivity = %.2f, want >= %.2f", got.DaysSinceActivity, tt.minDaysActivity)
			}
		})
	}
}

func TestComputeIssueStateThresholdsReported(t *testing.T) {
	raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(30)}
	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if got.StaleThresholdDays != 7 {
		t.Errorf("StaleThresholdDays = %v, want 7", got.StaleThresholdDays)
	}
	if got.CloseThresholdDays != 7 {
		t.Errorf("CloseThresholdDays = %v, want 7", got.CloseThresholdDays)
	}
	if got.IssueAuthor != testAuthor {
		t.Errorf("IssueAuthor = %q, want %q", got.IssueAuthor, testAuthor)
	}
}

func TestIsIgnoredActor(t *testing.T) {
	tests := []struct {
		login, self string
		want        bool
	}{
		{"", "x[bot]", true},
		{"github-actions[bot]", "x[bot]", true},
		{"x[bot]", "x[bot]", true},
		{"realuser", "x[bot]", false},
		{"realuser", "", false},
	}
	for _, tt := range tests {
		if got := isIgnoredActor(tt.login, tt.self); got != tt.want {
			t.Errorf("isIgnoredActor(%q,%q) = %v, want %v", tt.login, tt.self, got, tt.want)
		}
	}
}

// --- helpers to build raw GraphQL node slices --------------------------------

func commentNodes(cs ...rawComment) struct {
	Nodes []rawComment `json:"nodes"`
} {
	return struct {
		Nodes []rawComment `json:"nodes"`
	}{Nodes: cs}
}

func editNodes(es ...rawEdit) struct {
	Nodes []rawEdit `json:"nodes"`
} {
	return struct {
		Nodes []rawEdit `json:"nodes"`
	}{Nodes: es}
}

func timelineNodes(ts ...rawTimelineItem) struct {
	Nodes []rawTimelineItem `json:"nodes"`
} {
	return struct {
		Nodes []rawTimelineItem `json:"nodes"`
	}{Nodes: ts}
}

func labelNodes(names ...string) struct {
	Nodes []struct {
		Name string `json:"name"`
	} `json:"nodes"`
} {
	var nodes []struct {
		Name string `json:"name"`
	}
	for _, n := range names {
		nodes = append(nodes, struct {
			Name string `json:"name"`
		}{Name: n})
	}
	return struct {
		Nodes []struct {
			Name string `json:"name"`
		} `json:"nodes"`
	}{Nodes: nodes}
}

// --- TDD: review-driven fixes --------------------------------------------

// The maintainer's last ACTION is a non-comment (title rename), but the most
// recent COMMENT is the author's. STEP 3 (maintainer intent) must not analyze
// the author's words as the maintainer's, so LastCommentText must be empty.
func TestComputeIssueState_LastCommentAttributedToLastActor(t *testing.T) {
	raw := &rawIssue{
		Author:    actor(testAuthor),
		CreatedAt: daysAgo(30),
		Comments: commentNodes(
			rawComment{Author: actor(testAuthor), Body: "Could you please help me?", CreatedAt: daysAgo(10)},
		),
		TimelineItems: timelineNodes(
			rawTimelineItem{Typename: "RenamedTitleEvent", Actor: actor("maintainerA"), CreatedAt: daysAgo(9)},
		),
	}
	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if got.LastActionRole != string(roleMaintainer) {
		t.Fatalf("LastActionRole = %q, want maintainer", got.LastActionRole)
	}
	if got.LastCommentText != "" {
		t.Errorf("LastCommentText = %q, want empty (the author's comment must not be attributed to the maintainer)", got.LastCommentText)
	}
}

// A stale issue whose stale LabeledEvent has scrolled out of the bounded
// timeline window must still be closable: DaysSinceStaleLabel should fall back
// to time since last activity instead of reporting 0 (which never exceeds the
// close threshold).
func TestComputeIssueState_ClosableWhenStaleLabelEventOutOfWindow(t *testing.T) {
	raw := &rawIssue{
		Author:    actor(testAuthor),
		CreatedAt: daysAgo(60),
		Labels:    labelNodes(testStaleLabel),
		Comments: commentNodes(
			rawComment{Author: actor("maintainerA"), Body: "Marking stale.", CreatedAt: daysAgo(30)},
		),
		// No LabeledEvent for the stale label present in the timeline window.
	}
	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if !got.IsStale {
		t.Fatal("IsStale = false, want true")
	}
	if got.DaysSinceStaleLabel < got.CloseThresholdDays {
		t.Errorf("DaysSinceStaleLabel = %.2f, want >= close threshold %.2f so the issue can be closed",
			got.DaysSinceStaleLabel, got.CloseThresholdDays)
	}
}

// Guard: when the stale LabeledEvent IS in the window, DaysSinceStaleLabel is
// measured from that event.
func TestComputeIssueState_DaysSinceStaleLabelFromEvent(t *testing.T) {
	raw := &rawIssue{
		Author:    actor(testAuthor),
		CreatedAt: daysAgo(30),
		Labels:    labelNodes(testStaleLabel),
		TimelineItems: timelineNodes(
			rawTimelineItem{Typename: "LabeledEvent", CreatedAt: daysAgo(10), Label: &struct {
				Name string `json:"name"`
			}{Name: testStaleLabel}},
		),
	}
	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if got.DaysSinceStaleLabel < 9 || got.DaysSinceStaleLabel > 11 {
		t.Errorf("DaysSinceStaleLabel = %.2f, want ~10", got.DaysSinceStaleLabel)
	}
}
