// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Projects the user explicitly created via "New Project", tracked client-side
// so an EMPTY project still shows — without a backend registry, and without
// surfacing every incidental /workspace folder as a project.
const KEY = "lha.projects.created";

export function loadCreatedProjects(): Set<string> {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw);
    return Array.isArray(arr)
      ? new Set(arr.filter((x): x is string => typeof x === "string"))
      : new Set();
  } catch {
    return new Set();
  }
}

function persist(set: Set<string>): void {
  try {
    localStorage.setItem(KEY, JSON.stringify([...set]));
  } catch {
    // best-effort
  }
}

export function addCreatedProject(name: string): Set<string> {
  const set = loadCreatedProjects();
  set.add(name);
  persist(set);
  return set;
}

export function removeCreatedProject(name: string): Set<string> {
  const set = loadCreatedProjects();
  set.delete(name);
  persist(set);
  return set;
}
