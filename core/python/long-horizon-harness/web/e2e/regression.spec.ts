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

test("plain text streaming renders end-to-end", async ({ page }) => {
  await mockA2A(page, { streams: ["text-stream.jsonl"] });

  await page.goto("/");
  await page.getByRole("textbox").fill("hello");
  await page.keyboard.press("Enter");

  // The streamed reply must end up visible. The deltas merge into "Hello, world"
  // and the final non-partial event is dropped (text already streamed).
  await expect(page.getByText(/Hello, world/)).toBeVisible({ timeout: 10_000 });
});
