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
	"strings"
	"testing"
)

// An empty MAINTAINERS list silently disables stale-marking (no comment can be
// classified as a maintainer action), so the bot must surface a warning.
func TestMaintainersWarning(t *testing.T) {
	if w := maintainersWarning(&Config{Maintainers: nil}); w == "" {
		t.Error("expected a warning when MAINTAINERS is empty")
	}
	if w := maintainersWarning(&Config{Maintainers: []string{"alice"}}); w != "" {
		t.Errorf("expected no warning when maintainers are configured, got %q", w)
	}
}

func TestSummarize(t *testing.T) {
	if got := summarize("line one\nline two"); got != "line one line two" {
		t.Errorf("summarize collapsed newlines wrong: %q", got)
	}
	long := strings.Repeat("x", 500)
	if got := summarize(long); len(got) > 210 || !strings.HasSuffix(got, "...") {
		t.Errorf("summarize did not truncate long text: len=%d", len(got))
	}
}
