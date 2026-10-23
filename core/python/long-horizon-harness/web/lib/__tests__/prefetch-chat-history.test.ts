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

import { describe, it, expect, beforeEach, vi } from "vitest";
import type { Task } from "@a2a-js/sdk";
import {
  _resetTaskHistoryCache,
  getTaskHistoryCache,
} from "@/lib/task-history-cache";
import { queryClient } from "@/lib/query-client";
import { qk } from "@/lib/query-keys";
import type { HorizonClient } from "@/lib/a2a-client";

const tasksData = [
  { id: "t1", status: "completed", isActive: false },
  { id: "t2", status: "working", isActive: true },
];

vi.mock("@/lib/horizon-tasks", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/horizon-tasks")>();
  return {
    ...actual,
    fetcher: vi.fn(async () => tasksData),
  };
});

import { prefetchChatHistory } from "@/lib/prefetch-chat-history";

beforeEach(() => {
  _resetTaskHistoryCache();
  queryClient.clear();
});

describe("prefetchChatHistory", () => {
  it("warms the TanStack tasks cache and the task-history cache (terminal only)", async () => {
    const getTask = vi.fn(
      async (id: string) =>
        ({ id, status: { state: "completed" } }) as unknown as Task,
    );
    const client = { getTask } as unknown as HorizonClient;

    await prefetchChatHistory("ctx", client);

    // Warms the same cache entry useLhaTasks reads.
    expect(queryClient.getQueryData(qk.tasks("ctx"))).toEqual(tasksData);

    expect(getTask).toHaveBeenCalledTimes(1); // only the completed task
    expect(getTask).toHaveBeenCalledWith("t1");
    const cache = getTaskHistoryCache("ctx");
    expect(cache.get("t1")).toBeDefined();
    expect(cache.get("t2")).toBeUndefined(); // in-flight task not cached
  });

  it("no-ops without a client", async () => {
    await prefetchChatHistory("ctx", null);
    expect(getTaskHistoryCache("ctx").size).toBe(0);
    expect(queryClient.getQueryData(qk.tasks("ctx"))).toBeUndefined();
  });
});
