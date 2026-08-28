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
	"strings"
	"sync"
	"testing"
	"time"
)

// staleReady is an observation that satisfies the mark-stale preconditions.
func staleReady() IssueState {
	return IssueState{
		Status: "success", LastActionRole: string(roleMaintainer),
		IsStale: false, DaysSinceActivity: 30, StaleThresholdDays: 14, CloseThresholdDays: 7,
	}
}

// The label writes are idempotent but the comments are not, so a second
// emission of the same tool in one turn must not reach the API.
func TestClaimRefusesRepeatOfSameAction(t *testing.T) {
	c := newTestClient(t)
	// Dry-run so the write short-circuits before the (nil) REST transport: this
	// test is about the gate, not about the API call.
	c.cfg.DryRun = true
	c.recordObservation(7, staleReady())
	ctx := withAuditedIssue(context.Background(), 7)

	if res, err := c.doMarkStale(ctx, 7); err != nil || res.Status != "success" {
		t.Fatalf("first doMarkStale = (%+v, %v), want success", res, err)
	}

	res, err := c.doMarkStale(ctx, 7)
	if err != nil {
		t.Fatalf("second doMarkStale returned a Go error: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "already performed") {
		t.Fatalf("second doMarkStale = %+v, want a refusal naming the prior action", res)
	}
}

// STEP 1 legitimately removes the stale label AND posts an edit alert off a
// single get_issue_state, and STEP 3 marks stale AND adds the clarification
// label. A claim keyed only on the issue would refuse the second, correct call.
func TestClaimIsPerActionNotPerIssue(t *testing.T) {
	c := newTestClient(t)
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleAuthor)
	st.MaintainerAlertNeeded = true
	c.recordObservation(7, st)

	if msg, ok := c.claimAction(7, actionRemoveStale, removeStalePredicate(7)); !ok {
		t.Fatalf("removing the stale label was refused: %s", msg)
	}
	if msg, ok := c.claimAction(7, actionAlertEdit, alertPredicate(7)); !ok {
		t.Fatalf("the follow-up maintainer alert was refused: %s", msg)
	}
}

// Two goroutines racing the same action on the same issue: exactly one may win.
func TestClaimIsAtomicUnderConcurrency(t *testing.T) {
	c := newTestClient(t)
	c.recordObservation(7, staleReady())

	const goroutines = 8
	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		wins int
	)
	start := make(chan struct{})
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if _, ok := c.claimAction(7, actionMarkStale, stalePredicate(7)); ok {
				mu.Lock()
				wins++
				mu.Unlock()
			}
		}()
	}
	close(start)
	wg.Wait()
	if wins != 1 {
		t.Errorf("%d goroutines claimed the same action, want exactly 1", wins)
	}
}

// Stripping the stale label resets days_since_stale_label to zero, so an issue
// steered down that path can never reach the close branch. STEP 1 allows it only
// when the author or another user came back.
func TestRemoveStaleLabelRefusedWhenMaintainerActedLast(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleMaintainer)
	c.recordObservation(7, st)

	res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c.cfg.StaleLabel)
	if err != nil {
		t.Fatalf("doRemoveLabel returned a Go error: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "must stay") {
		t.Errorf("doRemoveLabel = %+v, want a refusal", res)
	}
}

// The alert posts a comment, so it needs the same Go-side gate as the rest.
func TestAlertEditRefusedWhenNotNeeded(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.MaintainerAlertNeeded = false
	c.recordObservation(7, st)

	res, err := c.doAlertEdit(withAuditedIssue(context.Background(), 7), 7)
	if err != nil {
		t.Fatalf("doAlertEdit returned a Go error: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "does not need a maintainer alert") {
		t.Errorf("doAlertEdit = %+v, want a refusal", res)
	}
}

// Editing an old comment is activity, but it does not change who spoke last.
// Attributing authorship to the editor let an author who tweaks a months-old
// comment displace a maintainer who replied in between.
func TestEditedCommentDoesNotDisplaceLastActor(t *testing.T) {
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	t1, t2 := t0.Add(24*time.Hour), t0.Add(48*time.Hour)
	st := replay([]historyEvent{
		{Type: eventCommented, Actor: "author", Time: t0, Body: "original"},
		{Type: eventCommented, Actor: "maint", Time: t1, Body: "can you share a repro?"},
		{Type: eventEditedComment, Actor: "author", Time: t2, Body: "original, tweaked"},
	}, toSet([]string{"maint"}), "author")

	if st.LastActorRole != roleMaintainer {
		t.Errorf("LastActorRole = %v, want %v (the maintainer spoke last)", st.LastActorRole, roleMaintainer)
	}
	if !st.LastActivity.Equal(t2) {
		t.Errorf("LastActivity = %v, want %v (the edit is still activity)", st.LastActivity, t2)
	}
	if st.LastCommentText != "can you share a repro?" {
		t.Errorf("LastCommentText = %q, want the maintainer's comment", st.LastCommentText)
	}
}

// An edit by the person who genuinely spoke last must still leave them as the
// last actor -- the fix must not over-correct.
func TestEditedCommentKeepsAuthorWhenTheySpokeLast(t *testing.T) {
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	st := replay([]historyEvent{
		{Type: eventCommented, Actor: "maint", Time: t0, Body: "repro?"},
		{Type: eventCommented, Actor: "author", Time: t0.Add(time.Hour), Body: "here it is"},
		{Type: eventEditedComment, Actor: "author", Time: t0.Add(2 * time.Hour), Body: "here it is, fixed"},
	}, toSet([]string{"maint"}), "author")

	if st.LastActorRole != roleAuthor {
		t.Errorf("LastActorRole = %v, want %v", st.LastActorRole, roleAuthor)
	}
	if !st.LastActivity.Equal(t0.Add(2 * time.Hour)) {
		t.Errorf("LastActivity = %v, want the edit time", st.LastActivity)
	}
}
