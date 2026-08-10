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

import { describe, it, expect } from "vitest";
import {
  matchesQuery,
  filterEntries,
  isHiddenName,
} from "@/components/chat/workspace-tree-filter";

type Entry = { name: string; kind: "file" | "dir" | "symlink" };

const entries: Entry[] = [
  { name: "report.pdf", kind: "file" },
  { name: "Notes.txt", kind: "file" },
  { name: "data", kind: "dir" },
];

describe("matchesQuery", () => {
  it("is case-insensitive substring on the basename", () => {
    expect(matchesQuery("Report.PDF", "report")).toBe(true);
    expect(matchesQuery("report.pdf", "PDF")).toBe(true);
    expect(matchesQuery("report.pdf", "xls")).toBe(false);
  });

  it("treats an empty/whitespace query as a match (identity)", () => {
    expect(matchesQuery("anything", "")).toBe(true);
    expect(matchesQuery("anything", "   ")).toBe(true);
  });
});

describe("isHiddenName", () => {
  it("flags dotfiles and dotfolders", () => {
    expect(isHiddenName(".lha")).toBe(true);
    expect(isHiddenName(".npm-global")).toBe(true);
    expect(isHiddenName(".")).toBe(true);
  });

  it("leaves normal names alone", () => {
    expect(isHiddenName("lha")).toBe(false);
    expect(isHiddenName("report.pdf")).toBe(false);
    expect(isHiddenName("a.b.c")).toBe(false);
  });
});

describe("filterEntries", () => {
  it("returns all entries for an empty query", () => {
    expect(filterEntries(entries, "")).toHaveLength(3);
  });

  it("keeps only name-matching entries for a non-empty query", () => {
    const out = filterEntries(entries, "note");
    expect(out.map((e) => e.name)).toEqual(["Notes.txt"]);
  });

  it("keeps a directory when its name matches even if children are unknown", () => {
    const out = filterEntries(entries, "data");
    expect(out.map((e) => e.name)).toEqual(["data"]);
  });
});
