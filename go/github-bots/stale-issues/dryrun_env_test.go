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

// DRY_RUN is the ONLY channel the reusable workflow uses to request a dry run:
// it sets the environment variable and never passes the -dry-run flag. The rest
// of the config suite asserts dry-run through the flag, so severing the env var
// from the flag default leaves those tests green while a caller's
// `dry_run: true` silently mutates live issues. This test pins the env path.
func TestLoadConfigDryRunFromEnv(t *testing.T) {
	for _, tc := range []struct {
		env  string
		want bool
	}{
		{"true", true},
		{"false", false},
		{"", false},
	} {
		t.Run("DRY_RUN="+tc.env, func(t *testing.T) {
			setRequiredCreds(t)
			t.Setenv("DRY_RUN", tc.env)
			cfg, err := loadConfig(nil)
			if err != nil {
				t.Fatalf("loadConfig: %v", err)
			}
			if cfg.DryRun != tc.want {
				t.Errorf("DRY_RUN=%q gave DryRun=%v, want %v", tc.env, cfg.DryRun, tc.want)
			}
		})
	}
}
