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

import { describe, expect, it } from "vitest";

import { buildMessageParts } from "@/lib/a2a-client";

describe("buildMessageParts", () => {
  it("omits the text part when text is empty (file with no caption)", () => {
    const file = {
      kind: "file" as const,
      file: { bytes: "AAAA", mimeType: "image/png", name: "a.png" },
    };
    const parts = buildMessageParts("", [file]);
    expect(parts).toEqual([file]);
    expect(parts.some((p) => p.kind === "text")).toBe(false);
  });

  it("omits the text part when text is whitespace only", () => {
    const file = {
      kind: "file" as const,
      file: { bytes: "AAAA", mimeType: "image/png", name: "a.png" },
    };
    expect(buildMessageParts("   \n", [file])).toEqual([file]);
  });

  it("keeps the text part (with original text) when it has content", () => {
    const parts = buildMessageParts("hello", []);
    expect(parts).toEqual([{ kind: "text", text: "hello" }]);
  });

  it("orders text before extra parts", () => {
    const file = {
      kind: "file" as const,
      file: { bytes: "AAAA", mimeType: "image/png", name: "a.png" },
    };
    const parts = buildMessageParts("caption", [file]);
    expect(parts).toEqual([{ kind: "text", text: "caption" }, file]);
  });

  it("returns an empty array when there is nothing to send", () => {
    expect(buildMessageParts("", [])).toEqual([]);
    expect(buildMessageParts("")).toEqual([]);
  });
});
