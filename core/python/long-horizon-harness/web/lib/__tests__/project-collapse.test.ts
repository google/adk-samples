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
import { loadCollapsed, persistCollapsed } from "@/lib/project-collapse";

// Node 26 ships a built-in localStorage that shadows happy-dom's; stub a
// Map-backed Storage so the module's logic is what's under test.
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

describe("project-collapse", () => {
  it("round-trips the collapsed set", () => {
    persistCollapsed(new Set(["a", "b"]));
    expect(loadCollapsed()).toEqual(new Set(["a", "b"]));
  });

  it("returns an empty set when nothing is stored", () => {
    expect(loadCollapsed()).toEqual(new Set());
  });

  it("tolerates corrupt JSON", () => {
    localStorage.setItem("lha.projects.collapsed", "{not json");
    expect(loadCollapsed()).toEqual(new Set());
  });
});
