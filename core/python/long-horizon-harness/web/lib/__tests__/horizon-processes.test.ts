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
import { hasRunning } from "@/lib/horizon-processes";

describe("hasRunning", () => {
  it("is false for undefined/empty", () => {
    expect(hasRunning(undefined)).toBe(false);
    expect(hasRunning({ processes: [] })).toBe(false);
  });

  it("is true when any process is running", () => {
    expect(
      hasRunning({
        processes: [
          {
            session_id: "a",
            command: "x",
            running: false,
            exit_code: 0,
            idle_seconds: 1,
            output_size: 0,
            pid: 1,
            started_at: 0,
          },
          {
            session_id: "b",
            command: "y",
            running: true,
            exit_code: null,
            idle_seconds: 1,
            output_size: 0,
            pid: 2,
            started_at: 0,
          },
        ],
      }),
    ).toBe(true);
  });

  it("is false when all exited", () => {
    expect(
      hasRunning({
        processes: [
          {
            session_id: "a",
            command: "x",
            running: false,
            exit_code: 0,
            idle_seconds: 1,
            output_size: 0,
            pid: 1,
            started_at: 0,
          },
        ],
      }),
    ).toBe(false);
  });
});
