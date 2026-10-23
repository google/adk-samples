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

test("clarify card renders question + choices and sends a continuation on click", async ({
  page,
}) => {
  const captured: CapturedRpc[] = [];
  await mockA2A(page, {
    streams: ["clarify-flow.jsonl", "clarify-continuation.jsonl"],
    captured,
  });

  await page.goto("/");
  // Empty state: type into the textarea and hit Send.
  const input = page.getByRole("textbox");
  await input.fill("create a plot of stock prices");
  await page.keyboard.press("Enter");

  // The clarify card surfaces the question and the three choice buttons.
  await expect(
    page.getByText("What kind of plot would you like?"),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "line" })).toBeVisible();
  await expect(page.getByRole("button", { name: "bar" })).toBeVisible();
  await expect(page.getByRole("button", { name: "scatter" })).toBeVisible();

  // Click a choice — this should fire sendConfirmation, which goes out as a
  // second message/stream RPC with the function_response DataPart.
  await page.getByRole("button", { name: "bar" }).click();

  // Confirm the continuation reply is rendered.
  await expect(page.getByText("Done!")).toBeVisible({ timeout: 10_000 });

  // Two outgoing stream calls: the initial prompt + the confirmation.
  const streamCalls = captured.filter(
    (c) => c.method === "SendStreamingMessage",
  );
  expect(streamCalls.length).toBeGreaterThanOrEqual(2);

  // The second call's message carries the function_response DataPart that
  // sendConfirmation built. (Sanity-check the wire shape end-to-end.)
  const confirmCall = streamCalls[streamCalls.length - 1];
  const params = confirmCall.params as {
    message: { parts: { data?: unknown; metadata?: { adk_type?: string } }[] };
  };
  const dataPart = params.message.parts.find(
    (p) => p.data !== undefined && p.metadata?.adk_type === "function_response",
  );
  expect(dataPart).toBeDefined();
  expect(dataPart?.data).toMatchObject({
    response: { confirmed: true, payload: { choice: "bar" } },
  });
});
