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

import { expect, test } from "@playwright/test";

// Opt-in only — runs the chat UI against whatever real backend
// `web/vite.config.ts` is proxying `/lha`, `/a2a`, and `/.well-known` to
// (typically the developer's :8001 ADK server). Off by default to keep
// `npm run test:e2e` hermetic.
//
// Enable: SMOKE_LIVE=1 npm run test:e2e -- smoke-against-live
test.describe("live backend smoke", () => {
  test.skip(!process.env.SMOKE_LIVE, "set SMOKE_LIVE=1 to enable");

  test("user can send a terminal command and see a response", async ({ page }) => {
    await page.goto("/");

    const composer = page.getByRole("textbox");
    await composer.fill(
      "Use the terminal tool to run exactly: echo live-smoke-marker",
    );
    await page.keyboard.press("Enter");

    // `live-smoke-marker` is the deterministic stdout from the agent's
    // `terminal` tool call — its presence in the chat thread proves the
    // round-trip (UI → /a2a → agent → terminal → response) worked without
    // asserting on LLM-generated prose.
    await expect(page.getByText(/live-smoke-marker/)).toBeVisible({
      timeout: 60_000,
    });
  });
});
