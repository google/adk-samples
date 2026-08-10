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

import { test, expect } from "@playwright/test";
import { mockA2A, type CapturedRpc } from "./helpers/mock-a2a";

test("confirmation card shows hint + originalCall and sends approve/decline", async ({
  page,
}) => {
  const captured: CapturedRpc[] = [];
  await mockA2A(page, {
    streams: ["confirmation-flow.jsonl", "clarify-continuation.jsonl"],
    captured,
  });

  await page.goto("/");
  await page.getByRole("textbox").fill("ls /tmp");
  await page.keyboard.press("Enter");

  await expect(page.getByText("Run shell command: ls /tmp ?")).toBeVisible();
  // The originalCall pill renders the tool name.
  await expect(page.getByText("terminal")).toBeVisible();

  const approve = page.getByRole("button", { name: /approve/i });
  const decline = page.getByRole("button", { name: /decline/i });
  await expect(approve).toBeVisible();
  await expect(decline).toBeVisible();

  await approve.click();

  await expect(page.getByText("Done!")).toBeVisible({ timeout: 10_000 });

  // A2A 1.0: the JSON-RPC method is SendStreamingMessage, and a data part is
  // serialised with a bare `data` key rather than a `kind` discriminator.
  const streamCalls = captured.filter((c) => c.method === "SendStreamingMessage");
  const confirmCall = streamCalls[streamCalls.length - 1];
  const params = confirmCall.params as {
    message: {
      parts: {
        data?: { response?: { confirmed?: boolean; payload?: unknown } };
        metadata?: { adk_type?: string };
      }[];
    };
  };
  const dataPart = params.message.parts.find(
    (p) => p.data !== undefined && p.metadata?.adk_type === "function_response",
  );
  expect(dataPart?.data?.response?.confirmed).toBe(true);
});
