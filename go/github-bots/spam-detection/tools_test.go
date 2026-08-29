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
	"sync/atomic"
	"testing"
)

// countingHandler writes a fixed body for every request and counts the calls.
func countingHandler(calls *int, status int, body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		*calls++
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

// flagHandler returns valid per-endpoint bodies (an array for labels, an object
// for the comment) and counts the calls.
func flagHandler(calls *int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		*calls++
		if strings.HasSuffix(r.URL.Path, "/comments") {
			_, _ = io.WriteString(w, `{"id":1}`)
			return
		}
		_, _ = io.WriteString(w, `[{"name":"spam"}]`)
	}
}

func TestFlagAsSpamRejectsWrongIssueWithoutHTTP(t *testing.T) {
	var calls int
	c := testClient(t, testConfig(), countingHandler(&calls, http.StatusOK, `{}`))
	ctx := withAuditedIssue(context.Background(), 7)
	res, err := c.flagAsSpam(ctx, 8, "promo")
	if err != nil {
		t.Fatalf("flagAsSpam() unexpected Go error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("status = %q, want error (wrong issue)", res.Status)
	}
	if calls != 0 {
		t.Errorf("made %d HTTP calls, want 0", calls)
	}
}

func TestFlagAsSpamRejectsUnscopedSessionWithoutHTTP(t *testing.T) {
	var calls int
	c := testClient(t, testConfig(), countingHandler(&calls, http.StatusOK, `{}`))
	// No withAuditedIssue: nothing is authorized.
	res, err := c.flagAsSpam(context.Background(), 7, "promo")
	if err != nil {
		t.Fatalf("flagAsSpam() unexpected Go error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("status = %q, want error (no session scope)", res.Status)
	}
	if calls != 0 {
		t.Errorf("made %d HTTP calls, want 0", calls)
	}
}

func TestFlagAsSpamAuthorizedSucceeds(t *testing.T) {
	var calls int
	c := testClient(t, testConfig(), flagHandler(&calls))
	ctx := withAuditedIssue(context.Background(), 7)
	res, err := c.flagAsSpam(ctx, 7, "promo link to a shoe store")
	if err != nil {
		t.Fatalf("flagAsSpam() error = %v", err)
	}
	if res.Status != "success" {
		t.Errorf("status = %q, want success", res.Status)
	}
	if calls != 2 {
		t.Errorf("made %d HTTP calls, want 2 (label + comment)", calls)
	}
}

func TestFlagAsSpamRESTErrorIsGoErrorAndRecorded(t *testing.T) {
	// Infrastructure failures (non-2xx) must surface as a Go error AND be
	// recorded so the run fails loudly.
	var calls int
	c := testClient(t, testConfig(), countingHandler(&calls, http.StatusInternalServerError, `{"message":"boom"}`))
	if c.hadError() {
		t.Fatal("hadError() should start false")
	}
	ctx := withAuditedIssue(context.Background(), 7)
	if _, err := c.flagAsSpam(ctx, 7, "promo"); err == nil {
		t.Fatal("flagAsSpam() expected Go error on HTTP 500, got nil")
	}
	if !c.hadError() {
		t.Error("flagAsSpam() did not record the infrastructure error")
	}
}

func TestFlagAsSpamDryRunSucceedsWithoutHTTP(t *testing.T) {
	cfg := testConfig()
	cfg.DryRun = true
	var calls int
	c := testClient(t, cfg, countingHandler(&calls, http.StatusOK, `{}`))
	ctx := withAuditedIssue(context.Background(), 7)
	res, err := c.flagAsSpam(ctx, 7, "promo")
	if err != nil {
		t.Fatalf("flagAsSpam() dry-run error = %v", err)
	}
	if res.Status != "success" {
		t.Errorf("status = %q, want success", res.Status)
	}
	if calls != 0 {
		t.Errorf("dry-run made %d HTTP calls, want 0", calls)
	}
}

func TestFlagAsSpamSecondCallIsNoOp(t *testing.T) {
	// The model might emit the flag tool twice for one issue. The first call
	// labels + comments (2 HTTP calls); the second must be a no-op so it cannot
	// post a duplicate alert comment.
	var calls int
	c := testClient(t, testConfig(), flagHandler(&calls))
	ctx := withAuditedIssue(context.Background(), 7)

	if res, _ := c.flagAsSpam(ctx, 7, "promo"); res.Status != "success" {
		t.Fatalf("first flag status = %q, want success", res.Status)
	}
	res, err := c.flagAsSpam(ctx, 7, "promo again")
	if err != nil {
		t.Fatalf("second flagAsSpam() error = %v", err)
	}
	if res.Status != "success" {
		t.Errorf("second flag status = %q, want success (no-op)", res.Status)
	}
	if calls != 2 {
		t.Errorf("made %d HTTP calls, want 2 (second flag must not call the API)", calls)
	}
}

func TestFlagAsSpamConcurrentIsolation(t *testing.T) {
	// Reviews of different issues run concurrently against the shared client.
	// Each session is scoped to its own issue; a session must only ever flag its
	// own issue, never another in flight. Run under -race to catch data races on
	// the shared client. Each handler call returns a per-endpoint body.
	var calls int32
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		if strings.HasSuffix(r.URL.Path, "/comments") {
			_, _ = io.WriteString(w, `{"id":1}`)
			return
		}
		_, _ = io.WriteString(w, `[{"name":"spam"}]`)
	}))

	const n = 20
	var wg sync.WaitGroup
	for i := 1; i <= n; i++ {
		wg.Add(1)
		go func(issue int) {
			defer wg.Done()
			ctx := withAuditedIssue(context.Background(), issue)
			// Flagging a DIFFERENT issue than the session scope must be rejected
			// without any HTTP call.
			if res, _ := c.flagAsSpam(ctx, issue+1000, "cross"); res.Status != "error" {
				t.Errorf("issue %d: cross-issue flag status = %q, want error", issue, res.Status)
			}
			// Flagging its own issue must succeed.
			if res, _ := c.flagAsSpam(ctx, issue, "self"); res.Status != "success" {
				t.Errorf("issue %d: self flag status = %q, want success", issue, res.Status)
			}
		}(i)
	}
	wg.Wait()

	// Each of the n issues is flagged exactly once: 2 HTTP calls (label + comment).
	if got := atomic.LoadInt32(&calls); got != 2*n {
		t.Errorf("made %d HTTP calls, want %d (label+comment per issue, no cross-issue writes)", got, 2*n)
	}
}

func TestAuthorizeIssue(t *testing.T) {
	ctx := withAuditedIssue(context.Background(), 7)
	if _, ok := authorizeIssue(ctx, 7); !ok {
		t.Error("authorizeIssue(7) on session scoped to 7 = not ok, want ok")
	}
	if _, ok := authorizeIssue(ctx, 8); ok {
		t.Error("authorizeIssue(8) on session scoped to 7 = ok, want not ok")
	}
	if _, ok := authorizeIssue(context.Background(), 7); ok {
		t.Error("authorizeIssue on unscoped session = ok, want not ok")
	}
}

// After a failed flag write, a second call must report the failure rather than
// "already flagged" -- otherwise the model's transcript records an alert comment
// that was never posted.
//
// The failure is PRODUCED by a failing write, not installed by hand: an earlier
// version of this test called recordFlagFailure directly, which left the
// recording half unpinned and made half the assertions a setter/getter
// tautology.
func TestSecondFlagAfterFailureReportsTheFailure(t *testing.T) {
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"message":"boom"}`)
	}))
	ctx := withAuditedIssue(context.Background(), 5)

	if _, err := c.flagAsSpam(ctx, 5, "spam"); err == nil {
		t.Fatal("the first flagAsSpam must surface the write failure")
	}
	if !c.hadError() {
		t.Error("a failed write must be recorded so the run exits non-zero")
	}

	res, err := c.flagAsSpam(ctx, 5, "spam")
	if err != nil {
		t.Fatalf("the second flagAsSpam returned a Go error: %v", err)
	}
	if res.Status != "error" {
		t.Fatalf("second flagAsSpam = %+v, want an error rather than a false success", res)
	}
	if !strings.Contains(res.Message, "already failed") {
		t.Errorf("message should say the attempt failed, got %q", res.Message)
	}
}
