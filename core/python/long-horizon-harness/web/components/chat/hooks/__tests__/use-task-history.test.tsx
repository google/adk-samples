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

import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import type { Task } from "@a2a-js/sdk";
import { useTaskHistory } from "@/components/chat/hooks/use-task-history";
import { _resetTaskHistoryCache } from "@/lib/task-history-cache";
import type { HorizonClient } from "@/lib/a2a-client";
import type { HorizonTaskSummary } from "@/lib/horizon-tasks";

function tp(text: string) {
  return { content: { $case: "text", value: text }, metadata: undefined, filename: "", mediaType: "" };
}
function dp(data: unknown, meta?: Record<string, unknown>) {
  return { content: { $case: "data", value: data }, metadata: meta, filename: "", mediaType: "" };
}

function makeTask(id: string, state: string): Task {
  return {
    id,
    contextId: "ctx",
    status: { state },
    history: [
      { messageId: `${id}-u`, role: "user",  parts: [tp("hi")] },
      { messageId: `${id}-a`, role: "agent", parts: [tp("yo")] },
    ],
  } as unknown as Task;
}

// A task holding only a confirmation card (the agent's adk_request_confirmation
// function_call) with no answer echo — mirrors the input-required task ADK
// leaves behind when it resumes an approved call into a fresh task.
function makeCardTask(id: string, callId: string): Task {
  return {
    id,
    contextId: "ctx",
    status: { state: "completed" },
    history: [
      {
        role: "agent",
        messageId: `${id}-card`,
        parts: [
          dp({
            id: callId,
            name: "adk_request_confirmation",
            args: {
              originalFunctionCall: { id: "t1", name: "terminal", args: { command: "rm -rf x" } },
              toolConfirmation: { hint: "Run rm -rf?", payload: null },
            },
          }, { adk_type: "function_call" }),
        ],
      },
    ],
  } as unknown as Task;
}

// The follow-up task carrying only the user's approve echo for `callId`.
function makeAnswerTask(id: string, callId: string): Task {
  return {
    id,
    contextId: "ctx",
    status: { state: "completed" },
    history: [
      {
        role: "user",
        messageId: `${id}-echo`,
        parts: [
          dp({
            id: callId,
            name: "adk_request_confirmation",
            response: { confirmed: true, payload: null },
          }, { adk_type: "function_response" }),
        ],
      },
    ],
  } as unknown as Task;
}

// Task with the gated tool's spinning function_call (no response yet).
function makeGatedCallTask(id: string, callId: string): Task {
  return {
    id, contextId: "ctx", status: { state: "completed" },
    history: [
      {
        role: "agent", messageId: `${id}-call`,
        parts: [
          dp({ id: callId, name: "terminal", args: { command: "rm -rf x" } }, { adk_type: "function_call" }),
          dp({
            id: `${callId}-conf`, name: "adk_request_confirmation",
            args: { originalFunctionCall: { id: callId, name: "terminal", args: { command: "rm -rf x" } }, toolConfirmation: { hint: "ok?", payload: null } },
          }, { adk_type: "function_call" }),
        ],
      },
    ],
  } as unknown as Task;
}

// Follow-up task: approve echo + the resumed terminal function_response.
function makeResumedRunTask(id: string, callId: string): Task {
  return {
    id, contextId: "ctx", status: { state: "completed" },
    history: [
      { role: "user",  messageId: `${id}-echo`, parts: [dp({ id: `${callId}-conf`, name: "adk_request_confirmation", response: { confirmed: true, payload: null } }, { adk_type: "function_response" })] },
      { role: "agent", messageId: `${id}-resp`, parts: [dp({ id: callId, name: "terminal", response: { stdout: "done", exit_code: 0 } }, { adk_type: "function_response" })] },
    ],
  } as unknown as Task;
}

const summary = (id: string, status: HorizonTaskSummary["status"]): HorizonTaskSummary => ({
  id,
  status,
  isActive: status === "working",
});

beforeEach(() => _resetTaskHistoryCache());

describe("useTaskHistory caching", () => {
  it("serves a revisited chat's terminal tasks from cache without re-fetching", async () => {
    const getTask = vi.fn(async (id: string) => makeTask(id, "completed"));
    const client = { getTask } as unknown as HorizonClient;
    const tasks = [summary("t1", "completed")];

    const first = renderHook((p) => useTaskHistory(p), {
      initialProps: { contextId: "ctx", tasks, client },
    });
    await waitFor(() =>
      expect(first.result.current.historyMessages).toHaveLength(2),
    );
    expect(getTask).toHaveBeenCalledTimes(1);
    first.unmount();

    const second = renderHook((p) => useTaskHistory(p), {
      initialProps: { contextId: "ctx", tasks, client },
    });
    await waitFor(() =>
      expect(second.result.current.historyMessages).toHaveLength(2),
    );
    expect(getTask).toHaveBeenCalledTimes(1); // still 1 — served from cache
  });

  it("always re-fetches an in-flight (non-terminal) task", async () => {
    const getTask = vi.fn(async (id: string) => makeTask(id, "working"));
    const client = { getTask } as unknown as HorizonClient;
    const tasks = [summary("t1", "working")];

    const first = renderHook((p) => useTaskHistory(p), {
      initialProps: { contextId: "ctx2", tasks, client },
    });
    await waitFor(() => expect(getTask).toHaveBeenCalledTimes(1));
    first.unmount();

    const second = renderHook((p) => useTaskHistory(p), {
      initialProps: { contextId: "ctx2", tasks, client },
    });
    await waitFor(() => expect(getTask).toHaveBeenCalledTimes(2)); // not cached
    second.unmount();
  });
});

describe("useTaskHistory cross-task HITL resolution", () => {
  it("resolves a card whose approve echo landed in a separate task", async () => {
    const callId = "adk-x";
    const getTask = vi.fn(async (id: string) =>
      id === "tcard" ? makeCardTask(id, callId) : makeAnswerTask(id, callId),
    );
    const client = { getTask } as unknown as HorizonClient;
    const tasks = [summary("tcard", "completed"), summary("tanswer", "completed")];

    const { result } = renderHook((p) => useTaskHistory(p), {
      initialProps: { contextId: "ctx-split", tasks, client },
    });

    await waitFor(() => expect(result.current.historyMessages).not.toBeNull());
    const conf = (result.current.historyMessages ?? [])
      .flatMap((m) => m.segments)
      .find((s) => s.kind === "confirmation");
    expect(conf).toMatchObject({
      callId,
      answer: { confirmed: true, text: null },
    });
  });
});

describe("useTaskHistory cross-task tool-result resolution", () => {
  it("clears the spinner when an approved tool runs in a later task", async () => {
    const callId = "term-1";
    const getTask = vi.fn(async (id: string) =>
      id === "tcall" ? makeGatedCallTask(id, callId) : makeResumedRunTask(id, callId),
    );
    const client = { getTask } as unknown as HorizonClient;
    const tasks = [summary("tcall", "completed"), summary("trun", "completed")];

    const { result } = renderHook((p) => useTaskHistory(p), {
      initialProps: { contextId: "ctx-tool-split", tasks, client },
    });

    await waitFor(() => expect(result.current.historyMessages).not.toBeNull());
    const tool = (result.current.historyMessages ?? [])
      .flatMap((m) => m.segments)
      .find((s) => s.kind === "tool" && s.name === "terminal");
    expect(tool).toMatchObject({ hasResult: true });
  });
});
