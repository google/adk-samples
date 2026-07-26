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

import { describe, expect, it, vi, beforeEach } from "vitest";

const cancelTaskMock = vi.fn();

vi.mock("@a2a-js/sdk/client", () => ({
  A2AClient: class {
    cancelTask = cancelTaskMock;
  },
}));

beforeEach(() => {
  cancelTaskMock.mockReset();
  global.fetch = vi.fn(async () =>
    new Response(
      JSON.stringify({
        name: "lha",
        url: "http://test/a2a",
        capabilities: { extensions: [] },
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    ),
  ) as unknown as typeof fetch;
});

import { createLhaClient } from "@/lib/a2a-client";

describe("cancelTask", () => {
  it("calls the underlying A2AClient.cancelTask with the task id", async () => {
    cancelTaskMock.mockResolvedValue({ id: "task-1", kind: "task" });

    const c = await createLhaClient({ contextId: "ctx-test" });
    await c.cancelTask("task-1");

    expect(cancelTaskMock).toHaveBeenCalledTimes(1);
    expect(cancelTaskMock.mock.calls[0][0]).toEqual({ id: "task-1" });
  });

  it("propagates RPC errors so the caller can ignore them", async () => {
    cancelTaskMock.mockRejectedValue(new Error("Task cannot be canceled"));

    const c = await createLhaClient({ contextId: "ctx-test" });
    await expect(c.cancelTask("task-2")).rejects.toThrow("Task cannot be canceled");
  });
});
