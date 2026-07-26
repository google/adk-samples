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
import { extractTaskId } from "@/lib/horizon-events";

describe("extractTaskId", () => {
  it("reads taskId from a status-update event", () => {
    expect(
      extractTaskId({ kind: "status-update", taskId: "t1", final: false }),
    ).toBe("t1");
  });

  it("reads id from a task event", () => {
    expect(
      extractTaskId({ kind: "task", id: "t2", status: { state: "working" } }),
    ).toBe("t2");
  });

  it("reads taskId from a final message event", () => {
    expect(
      extractTaskId({ kind: "message", taskId: "t3", parts: [] }),
    ).toBe("t3");
  });

  it("returns null when no task id is present", () => {
    expect(extractTaskId({ kind: "message", parts: [] })).toBeNull();
    expect(extractTaskId(null)).toBeNull();
    expect(extractTaskId("nope")).toBeNull();
  });
});
