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

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createProjectFolder,
  deleteProject,
  groupSessionsByProject,
} from "@/lib/projects";
import type { HorizonSessionSummary } from "@/lib/horizon-sessions";

function s(
  id: string,
  workspaceWindow: string[] = [],
): HorizonSessionSummary {
  return {
    id,
    title: id,
    createdAt: 0,
    lastUpdated: 0,
    source: "user",
    jobType: null,
    workspaceWindow,
  };
}

describe("groupSessionsByProject", () => {
  it("groups chats by anchor and buckets unanchored into general", () => {
    const { projects, general } = groupSessionsByProject(
      [s("a", ["projA"]), s("b"), s("c", ["projA"])],
      ["projA"],
    );
    expect(general.map((x) => x.id)).toEqual(["b"]);
    expect(projects).toHaveLength(1);
    expect(projects[0].name).toBe("projA");
    expect(projects[0].chats.map((x) => x.id)).toEqual(["a", "c"]);
  });

  it("hides an empty folder unless the user created it", () => {
    expect(groupSessionsByProject([s("a")], ["empty"]).projects).toEqual([]);
    expect(
      groupSessionsByProject([s("a")], ["empty"], new Set(["empty"])).projects,
    ).toEqual([{ name: "empty", chats: [] }]);
  });

  it("hides system + hidden folders and buckets their chats into General", () => {
    const { projects, general } = groupSessionsByProject(
      [
        s("a", ["lha"]),
        s("b", [".secret"]),
        s("c", ["uploads"]),
        s("d", ["real"]),
      ],
      [".lha", "lha", "uploads", "real"],
    );
    expect(projects.map((p) => p.name)).toEqual(["real"]);
    expect(general.map((x) => x.id).sort()).toEqual(["a", "b", "c"]);
  });

  it("appends anchor-only names (no folder) after folders, alpha", () => {
    const { projects } = groupSessionsByProject(
      [s("a", ["zeta"]), s("b", ["alpha"]), s("c", ["folder1"])],
      ["folder1"],
    );
    expect(projects.map((p) => p.name)).toEqual(["folder1", "alpha", "zeta"]);
  });
});

describe("project folder API clients", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("createProjectFolder POSTs the name and returns the path", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ path: "/workspace/projA" }),
    })) as unknown as typeof fetch;
    vi.stubGlobal("fetch", fetchMock);

    const out = await createProjectFolder("projA");
    expect(out.path).toBe("/workspace/projA");
    const [url, init] = (fetchMock as unknown as ReturnType<typeof vi.fn>).mock
      .calls[0];
    expect(url).toBe("/lha/workspace/folder");
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      name: "projA",
    });
  });

  it("deleteProject deletes chats before the folder", async () => {
    const urls: string[] = [];
    const fetchMock = vi.fn(async (url: string) => {
      urls.push(url);
      return { ok: true, status: 200, json: async () => ({}) };
    }) as unknown as typeof fetch;
    vi.stubGlobal("fetch", fetchMock);

    await deleteProject("Foo", ["c1", "c2"]);

    expect(urls).toEqual([
      "/lha/sessions/c1",
      "/lha/sessions/c2",
      "/lha/workspace/file?path=%2Fworkspace%2FFoo&recursive=true",
    ]);
  });

  it("deleteProject tolerates a 404 on the folder (anchor-only project)", async () => {
    const fetchMock = vi.fn(async (url: string) => ({
      ok: url.includes("/workspace/file") ? false : true,
      status: url.includes("/workspace/file") ? 404 : 200,
      json: async () => ({}),
      text: async () => "not found",
    })) as unknown as typeof fetch;
    vi.stubGlobal("fetch", fetchMock);

    await expect(deleteProject("Foo", ["c1"])).resolves.toBeUndefined();
  });
});
