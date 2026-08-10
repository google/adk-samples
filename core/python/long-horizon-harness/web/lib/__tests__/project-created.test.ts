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

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  addCreatedProject,
  loadCreatedProjects,
  removeCreatedProject,
} from "@/lib/project-created";

beforeEach(() => {
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
    clear: () => store.clear(),
  });
});
afterEach(() => vi.unstubAllGlobals());

describe("project-created", () => {
  it("adds and loads created projects", () => {
    addCreatedProject("foo");
    expect(loadCreatedProjects()).toEqual(new Set(["foo"]));
  });

  it("removes a created project", () => {
    addCreatedProject("foo");
    addCreatedProject("bar");
    removeCreatedProject("foo");
    expect(loadCreatedProjects()).toEqual(new Set(["bar"]));
  });

  it("returns an empty set when nothing is stored", () => {
    expect(loadCreatedProjects()).toEqual(new Set());
  });
});
