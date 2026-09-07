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
import { mockA2A } from "./helpers/mock-a2a";

// A cold mount that already has a contextId (deep link, or `/` redirecting to
// the last chat) must still finish booting. Regression: the boot effect stamped
// bootedContextIdRef before the async client resolved, so StrictMode's
// mount->cleanup->mount aborted the first boot and the second returned early on
// the already-stamped ref — leaving the client null and the pill on "connecting".
test("cold mount with a contextId in the URL reaches ready", async ({ page }) => {
  await mockA2A(page, { streams: [] });

  await page.goto("/c?id=11111111-2222-3333-4444-555555555555");

  await expect(page.getByText("ready", { exact: true })).toBeVisible({
    timeout: 15_000,
  });
});

test("plain text streaming renders end-to-end", async ({ page }) => {
  await mockA2A(page, { streams: ["text-stream.jsonl"] });

  await page.goto("/");
  await page.getByRole("textbox").fill("hello");
  await page.keyboard.press("Enter");

  // The streamed reply must end up visible. The deltas merge into "Hello, world"
  // and the final non-partial event is dropped (text already streamed).
  await expect(page.getByText(/Hello, world/)).toBeVisible({ timeout: 10_000 });
});
