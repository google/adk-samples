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
	"io"
	"net/http"
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

// add_label_to_issue is not threshold-gated, so allowing it to apply the stale
// label would be a way around add_stale_label_and_comment. An issue would become
// is_stale with no warning comment and its close clock already running, letting a
// later run close it after CloseAfter days instead of StaleAfter + CloseAfter --
// with the author never warned. This is reachable by steering the model, which is
// the whole threat these bots operate under.
func TestAddLabelCannotApplyTheStaleLabel(t *testing.T) {
	c := newTestClient(t)
	c.cfg.RequestClarificationLabel = "request clarification"
	ctx := withAuditedIssue(context.Background(), 7)

	res, err := c.doAddLabel(ctx, 7, c.cfg.StaleLabel)
	if err != nil {
		t.Fatalf("doAddLabel returned a Go error: %v", err)
	}
	if res.Status != "error" {
		t.Fatalf("doAddLabel(7, %q) = %+v, want a refusal", c.cfg.StaleLabel, res)
	}
	if !strings.Contains(res.Message, "add_stale_label_and_comment") {
		t.Errorf("refusal should point at the gated tool, got %q", res.Message)
	}
	// Nothing was ever recorded for issue 7, so the refusal also does not depend
	// on get_issue_state having run first.
}

// The tool must still work for the label it is actually for, on an issue that
// satisfies the same precondition marking it stale would.
func TestAddLabelStillAppliesTheClarificationLabel(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	c.cfg.RequestClarificationLabel = "request clarification"
	c.recordObservation(7, staleReady())
	res, err := c.doAddLabel(withAuditedIssue(context.Background(), 7), 7, "request clarification")
	if err != nil || res.Status != "success" {
		t.Fatalf("doAddLabel = (%+v, %v), want success", res, err)
	}
}

// An unknown stale-label age must not be closed on. Substituting time since
// activity biased every such issue toward an early close, and the issue author
// can force that branch by padding the timeline window with renames and reopens.
func TestCloseRefusedWhenStaleLabelAgeIsUnknown(t *testing.T) {
	c := newTestClient(t)
	st := staleReady()
	st.IsStale = true
	st.DaysSinceStaleLabel = -1 // the LabeledEvent fell outside the window
	c.recordObservation(7, st)

	res, err := c.doClose(withAuditedIssue(context.Background(), 7), 7)
	if err != nil {
		t.Fatalf("doClose returned a Go error: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "unknown length of time") {
		t.Errorf("doClose = %+v, want a refusal on unknown label age", res)
	}
}

// Each issue must get its own fence marker, so a marker disclosed once cannot be
// used to close the fence on a later issue in the same run.
func TestFenceMarkerIsPerIssue(t *testing.T) {
	a, err := newNonce()
	if err != nil {
		t.Fatalf("newNonce: %v", err)
	}
	b, err := newNonce()
	if err != nil {
		t.Fatalf("newNonce: %v", err)
	}
	if a == b {
		t.Fatal("two draws produced the same marker")
	}
	if got := fenceUntrusted("text", a); !strings.Contains(got, a) || strings.Contains(got, b) {
		t.Errorf("fenceUntrusted used the wrong marker: %q", got)
	}
}

// GitHub label names are case-insensitively unique, so an exact-match refusal
// let "Stale" through while landing the real stale label -- reopening the
// ungated route this bot closed. Both the tool check and config validation must
// compare case-insensitively.
func TestStaleLabelRefusalIsCaseInsensitive(t *testing.T) {
	c := newTestClient(t)
	c.cfg.StaleLabel = "stale"
	c.cfg.RequestClarificationLabel = "Stale" // the colliding misconfiguration
	ctx := withAuditedIssue(context.Background(), 7)

	res, err := c.doAddLabel(ctx, 7, "Stale")
	if err != nil {
		t.Fatalf("doAddLabel returned a Go error: %v", err)
	}
	if res.Status != "error" {
		t.Fatalf("doAddLabel(7, \"Stale\") = %+v, want a refusal", res)
	}
}

func TestValidateRejectsCollidingLabelNames(t *testing.T) {
	cfg := &Config{
		GitHubToken: "t", GeminiAPIKey: "k", Owner: "o", Repo: "r",
		StaleAfter: 336, CloseAfter: 168, IssueTimeout: 1, Concurrency: 1,
		StaleLabel: "stale", RequestClarificationLabel: "Stale",
	}
	err := cfg.validate()
	if err == nil {
		t.Fatal("validate() = nil, want an error on colliding label names")
	}
	if !strings.Contains(err.Error(), "must differ") {
		t.Errorf("validate() = %v, want it to name the collision", err)
	}
	cfg.RequestClarificationLabel = "request clarification"
	if err := cfg.validate(); err != nil {
		t.Errorf("distinct label names must pass: %v", err)
	}
}

// STEP 3 marks stale and then adds the clarification label off ONE
// get_issue_state. Separate action keys are what let the second call through.
func TestMarkStaleThenAddClarificationBothSucceed(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	c.cfg.RequestClarificationLabel = "request clarification"
	c.recordObservation(7, staleReady())
	ctx := withAuditedIssue(context.Background(), 7)

	if res, err := c.doMarkStale(ctx, 7); err != nil || res.Status != "success" {
		t.Fatalf("doMarkStale = (%+v, %v), want success", res, err)
	}
	if res, err := c.doAddLabel(ctx, 7, "request clarification"); err != nil || res.Status != "success" {
		t.Fatalf("follow-up doAddLabel = (%+v, %v), want success", res, err)
	}
}

// The clarification label is added on a threshold-satisfying issue only, so an
// issue nobody is waiting on cannot be flagged as waiting on its author.
func TestAddClarificationRefusedBelowThreshold(t *testing.T) {
	c := newTestClient(t)
	c.cfg.RequestClarificationLabel = "request clarification"
	st := staleReady()
	st.DaysSinceActivity = 1 // well inside the threshold
	c.recordObservation(7, st)

	res, err := c.doAddLabel(withAuditedIssue(context.Background(), 7), 7, "request clarification")
	if err != nil {
		t.Fatalf("doAddLabel returned a Go error: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "waiting on its author") {
		t.Errorf("doAddLabel = %+v, want a refusal naming THIS action", res)
	}
	if !strings.Contains(res.Message, "stale threshold") {
		t.Errorf("the refusal should still carry the underlying reason, got %q", res.Message)
	}
}

// The decision tree removes only the stale label.
func TestRemoveRefusesAnyLabelButStale(t *testing.T) {
	c := newTestClient(t)
	c.cfg.RequestClarificationLabel = "request clarification"
	res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, "request clarification")
	if err != nil {
		t.Fatalf("doRemoveLabel returned a Go error: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "only removes") {
		t.Errorf("doRemoveLabel = %+v, want a refusal", res)
	}
}

// RemoveLabel's dry-run guard was the one mutation path in this module with no
// test: deleting shouldSkip from it left the whole suite green.
func TestDryRunSuppressesRemoveLabel(t *testing.T) {
	calls := 0
	cfg := baseCfg()
	cfg.DryRun = true
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `[]`)
	}))
	if err := c.RemoveLabel(context.Background(), 7, cfg.StaleLabel); err != nil {
		t.Fatalf("RemoveLabel in dry-run = %v, want nil", err)
	}
	if calls != 0 {
		t.Errorf("RemoveLabel made %d HTTP calls in dry-run, want 0", calls)
	}
}

// doGetIssueState is the ONLY production caller of recordObservation, and every
// other test installs observations by hand. Without this, deleting that call
// leaves the suite green while production silently stops recording — and every
// destructive tool then refuses, so the bot does nothing at all.
func TestGetIssueStateRecordsTheObservation(t *testing.T) {
	const body = `{"data":{"repository":{"issue":{
		"author":{"login":"author"},"createdAt":"2020-01-01T00:00:00Z",
		"labels":{"nodes":[]},"comments":{"nodes":[]},
		"userContentEdits":{"nodes":[]},"timelineItems":{"nodes":[]}}}}}`
	c := testClient(t, baseCfg(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	if _, err := c.doGetIssueState(withAuditedIssue(context.Background(), 7), 7); err != nil {
		t.Fatalf("doGetIssueState: %v", err)
	}
	c.mu.Lock()
	_, recorded := c.observed[7]
	c.mu.Unlock()
	if !recorded {
		t.Fatal("doGetIssueState did not record the observation the destructive tools gate on")
	}
}

// The fence marker must be drawn BEFORE the observation is recorded. Recording
// first leaves a passing observation behind on the error path, which a later
// destructive tool could claim against even though the model never saw the
// state — clearing the mechanical gate while skipping the judgement the prompt
// exists to make. That ordering is a named security invariant with no other test.
func TestObservationIsNotRecordedWhenTheFenceCannotBeBuilt(t *testing.T) {
	src := readSource(t, "tools.go")
	nonceAt := strings.Index(src, "nonce, err := newNonce()")
	recordAt := strings.Index(src, "c.recordObservation(number, st)")
	if nonceAt < 0 || recordAt < 0 {
		t.Fatal("could not locate the nonce draw and the observation record in tools.go")
	}
	if nonceAt > recordAt {
		t.Error("recordObservation runs before newNonce: a CSPRNG failure would leave a claimable observation behind")
	}
}

// The tool set is the bot's entire authority surface. Pin it, so a seventh tool
// cannot be added without a deliberate decision — an ungated tool reaching a
// gated state is exactly how the stale-label bypass happened.
func TestToolSetIsExactlyTheSixKnownTools(t *testing.T) {
	c := newTestClient(t)
	tools, err := c.tools()
	if err != nil {
		t.Fatalf("tools() error = %v", err)
	}
	got := make(map[string]bool, len(tools))
	for _, tl := range tools {
		got[tl.Name()] = true
	}
	want := []string{
		"get_issue_state", "add_label_to_issue", "remove_label_from_issue",
		"add_stale_label_and_comment", "alert_maintainer_of_edit", "close_as_stale",
	}
	for _, n := range want {
		if !got[n] {
			t.Errorf("missing tool %q", n)
		}
	}
	if len(tools) != len(want) {
		t.Errorf("got %d tools %v, want exactly %d — a new tool needs its own Go gate", len(tools), got, len(want))
	}
}
