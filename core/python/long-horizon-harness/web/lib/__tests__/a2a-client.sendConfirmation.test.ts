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

const sendMessageStreamMock = vi.fn();

vi.mock("@a2a-js/sdk/client", () => ({
  A2AClient: class {
    sendMessageStream = sendMessageStreamMock;
  },
}));

// Stub the agent-card fetch the createLhaClient bootstrap requires.
beforeEach(() => {
  sendMessageStreamMock.mockReset();
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

async function* yieldNothing() {
  // Empty event stream — the test only inspects the OUTGOING params.
}

describe("sendConfirmation", () => {
  it("builds a continuation Message carrying an ADK function_response DataPart", async () => {
    sendMessageStreamMock.mockReturnValue(yieldNothing());

    const c = await createLhaClient({
      contextId: "ctx-test",
    });

    const stream = c.sendConfirmation({
      callId: "adk-conf-1",
      confirmed: true,
      payload: { choice: "line" },
    });
    // Drain the (empty) iterator so the body runs.
    for await (const _ of stream) {
      // no-op
    }

    expect(sendMessageStreamMock).toHaveBeenCalledTimes(1);
    const [params] = sendMessageStreamMock.mock.calls[0];
    expect(params.message.kind).toBe("message");
    expect(params.message.role).toBe("user");
    expect(params.message.contextId).toBe("ctx-test");
    expect(params.message.parts).toHaveLength(1);
    const part = params.message.parts[0];
    // ADK's `_parse_tool_confirmation` expects the response dict to be the
    // ToolConfirmation fields directly — not wrapped under a `toolConfirmation`
    // key. See the matching comment in lib/a2a-client.ts.
    expect(part).toEqual({
      kind: "data",
      data: {
        id: "adk-conf-1",
        name: "adk_request_confirmation",
        response: {
          confirmed: true,
          payload: { choice: "line" },
        },
      },
      metadata: { adk_type: "function_response" },
    });
  });

  it("sends confirmed=false with null payload on Decline", async () => {
    sendMessageStreamMock.mockReturnValue(yieldNothing());

    const c = await createLhaClient({
      contextId: "ctx-test",
    });

    const stream = c.sendConfirmation({
      callId: "adk-conf-2",
      confirmed: false,
      payload: null,
    });
    for await (const _ of stream) {
      // no-op
    }

    const [params] = sendMessageStreamMock.mock.calls[0];
    const part = params.message.parts[0];
    expect(part.data.response).toEqual({
      confirmed: false,
      payload: null,
    });
  });

  it("uses the override toolName when provided", async () => {
    sendMessageStreamMock.mockReturnValue(yieldNothing());

    const c = await createLhaClient({
      contextId: "ctx-test",
    });

    const stream = c.sendConfirmation({
      callId: "adk-call-clarify-1",
      confirmed: true,
      payload: { choice: "bar" },
      toolName: "clarify",
    });
    for await (const _ of stream) {
      // no-op
    }

    const [params] = sendMessageStreamMock.mock.calls[0];
    expect(params.message.parts[0].data.name).toBe("clarify");
  });
});
