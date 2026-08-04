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

import { type Page, type Route, type Request } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";

const FIXTURES_DIR = path.resolve(__dirname, "../fixtures");

export interface CapturedRpc {
  method: string;
  params: unknown;
  body: unknown;
}

export interface MockA2AOptions {
  /**
   * Queue of stream fixtures to use for successive `message/stream` calls.
   * Each entry names a file under `fixtures/streams/`. The first call drains
   * the first entry, the second the second, etc.
   */
  streams: string[];
  /** Optional capture array — pushed to on every JSON-RPC request. */
  captured?: CapturedRpc[];
  /**
   * Optional lha state to return from /lha/state. Defaults to an
   * "empty" state object that satisfies the HorizonStateResponse shape.
   */
  lhaState?: Record<string, unknown>;
}

const DEFAULT_LHA_STATE = {
  contextId: "ctx-e2e",
  exists: true,
  state: {
    skill_telemetry: {},
    halt_reason: null,
    last_error: null,
    iteration: 0,
    tool_calls_this_iteration: 0,
    iteration_halted: false,
    iteration_halt_reason: null,
    session_started: true,
    session_started_at: null,
    pending_confirmation: null,
    pending_credential: null,
    tool_call_log: [],
    delegate_history: [],
    memory_writes: [],
    bound_skills: [],
    in_flight_tool_call: null,
    sandbox_status: null,
  },
};

function loadStream(name: string): string[] {
  const fp = path.join(FIXTURES_DIR, "streams", name);
  const text = readFileSync(fp, "utf8");
  return text.split("\n").filter((line) => line.trim().length > 0);
}

function buildSseBody(eventLines: string[], rpcId: number | string): string {
  // Each A2A event is wrapped in a JSON-RPC response and sent as one SSE
  // `data:` frame. Two newlines terminate the frame.
  return eventLines
    .map((line) => {
      const event = JSON.parse(line);
      const payload = JSON.stringify({
        jsonrpc: "2.0",
        id: rpcId,
        result: event,
      });
      return `data: ${payload}\n\n`;
    })
    .join("");
}

export async function mockA2A(page: Page, opts: MockA2AOptions): Promise<void> {
  const streamQueue = [...opts.streams];
  const lhaState = opts.lhaState ?? DEFAULT_LHA_STATE;
  const captured = opts.captured ?? [];
  const agentCard = JSON.parse(
    readFileSync(path.join(FIXTURES_DIR, "agent-card.json"), "utf8"),
  );

  await page.route("**/.well-known/agent-card.json", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(agentCard),
    });
  });

  await page.route("**/lha/state**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(lhaState),
    });
  });

  await page.route("**/lha/contexts/**/tasks**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([]),
    });
  });

  await page.route("**/lha/sessions**", async (route) => {
    const url = route.request().url();
    if (/\/sessions\/[^/?]+(?:\?|$)/.test(url)) {
      const method = route.request().method();
      if (method === "DELETE") {
        await route.fulfill({ status: 204, body: "" });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ id: "ctx-e2e", title: "Mock chat" }),
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([]),
    });
  });

  await page.route("**/a2a", async (route: Route) => {
    const req: Request = route.request();
    if (req.method() !== "POST") {
      await route.fallback();
      return;
    }
    let body: { method?: string; id?: string | number; params?: unknown } = {};
    try {
      body = JSON.parse(req.postData() ?? "{}");
    } catch {
      body = {};
    }
    captured.push({
      method: body.method ?? "",
      params: body.params,
      body,
    });
    const method = body.method ?? "";
    const rpcId = body.id ?? 1;

    // A2A 1.0 renamed the JSON-RPC methods (message/stream -> SendStreamingMessage,
    // tasks/get -> GetTask, tasks/cancel -> CancelTask, tasks/resubscribe ->
    // SubscribeToTask). Accept the 1.0 name; the v0.3 name is kept so a stray
    // legacy caller still resolves rather than silently draining the queue.
    if (method === "SendStreamingMessage" || method === "message/stream") {
      const fixture = streamQueue.shift();
      if (!fixture) {
        await route.fulfill({ status: 200, contentType: "text/event-stream", body: "" });
        return;
      }
      const lines = loadStream(fixture);
      const sse = buildSseBody(lines, rpcId);
      await route.fulfill({
        status: 200,
        headers: { "content-type": "text/event-stream", "cache-control": "no-cache" },
        body: sse,
      });
      return;
    }

    // Catch-all for tasks/get and friends — return an empty success result.
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ jsonrpc: "2.0", id: rpcId, result: {} }),
    });
  });
}
