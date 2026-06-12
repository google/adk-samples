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

// Command githubtriagebot is an autonomous agent that triages open GitHub
// issues for a repository.
//
// For each untriaged issue it sets the GitHub issue type (Bug, Feature, or
// Task) and applies one categorization label (e.g. "bug", "enhancement",
// "documentation", "question") drawn from a configurable allowlist. An issue is
// considered untriaged when it has no issue type and/or none of the allowlisted
// categorization labels.
//
// The sample demonstrates:
//   - building an llmagent.New agent driven by typed functiontool.New tools;
//   - running it headlessly with a runner.Runner and an in-memory session,
//     consuming the streaming iter.Seq2[*session.Event, error] response;
//   - calling the GitHub REST API (go-github) and GraphQL API (a raw POST
//     through the same authenticated client) from inside tools;
//   - a clean split between pure, table-tested decision logic (triage.go) and
//     side-effecting I/O (github.go).
//
// The deterministic facts (which issues need triaging) are computed in code;
// only the fuzzy classification (which type and label fit an issue) is
// delegated to the model. A -dry-run flag logs intended actions without
// mutating anything.
package main
