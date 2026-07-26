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

// Thin wrapper around @a2a-js/sdk that fetches the agent card and exposes
// sendMessageStream()/getTask()/resubscribeTask() over same-origin /a2a.
//
// Same-origin requests to FastAPI: agent card at /.well-known/agent-card.json,
// RPC at /a2a. In dev, the Vite proxy (vite.config.ts) forwards these to the
// backend on :8001; in production the web/server/ Express proxy forwards them
// to the backend.

import { A2AClient } from "@a2a-js/sdk/client";
import type {
  AgentCard,
  Message,
  MessageSendParams,
  Part,
  Task,
} from "@a2a-js/sdk";
import { v4 as uuid } from "uuid";

const RPC_PATH = "/a2a";

export interface HorizonClient {
  card: AgentCard;
  client: A2AClient;
  sendStream: (text: string, opts?: SendOptions) => AsyncGenerator<unknown, void, void>;
  sendConfirmation: (
    opts: SendConfirmationOptions,
  ) => AsyncGenerator<unknown, void, void>;
  sendConfirmations: (
    items: ConfirmationItem[],
    opts?: { signal?: AbortSignal; messageId?: string },
  ) => AsyncGenerator<unknown, void, void>;
  getTask: (taskId: string, opts?: { signal?: AbortSignal }) => Promise<Task>;
  cancelTask: (taskId: string, opts?: { signal?: AbortSignal }) => Promise<void>;
  resubscribeTask: (
    taskId: string,
    opts?: { signal?: AbortSignal },
  ) => AsyncGenerator<unknown, void, void>;
  contextId: string;
}

export interface SendConfirmationOptions {
  /** The function_call id the agent is awaiting a response for. */
  callId: string;
  /** True if the user approved (clarify choice picked or Approve clicked). */
  confirmed: boolean;
  /**
   * For clarify-style confirmations, the picked choice or free-form answer.
   * For yes/no confirmations, any extra payload the original confirmation
   * carried (or null).
   */
  payload?: Record<string, unknown> | null;
  signal?: AbortSignal;
  messageId?: string;
  /** Optional original tool name for the function_response envelope. */
  toolName?: string;
}

export interface ConfirmationItem {
  callId: string;
  confirmed: boolean;
  payload?: Record<string, unknown> | null;
  toolName?: string;
}

export function buildConfirmationDataPart(item: ConfirmationItem): Part {
  return {
    kind: "data",
    data: {
      id: item.callId,
      name: item.toolName ?? "adk_request_confirmation",
      // ADK's `_parse_tool_confirmation` expects the ToolConfirmation fields
      // directly (confirmed/payload), not wrapped under a `toolConfirmation` key.
      response: {
        confirmed: item.confirmed,
        payload: item.payload ?? null,
      },
    },
    metadata: { adk_type: "function_response" },
  };
}

export interface CreateLhaClientOptions {
  contextId: string;
  /** Aborts the agent-card retry loop (e.g. when the boot effect tears down). */
  signal?: AbortSignal;
}

export interface SendOptions {
  /** Override the auto-generated message id. */
  messageId?: string;
  /** Abort signal forwarded to the underlying SDK request. */
  signal?: AbortSignal;
  /** Additional Parts appended after the text part (e.g., attachment files). */
  extraParts?: Part[];
}

function formatRpcError(err: { code?: number; message?: string }): string {
  return `${err.code ?? "?"} ${err.message ?? "no message"}`;
}

// Per-attempt cap. The backend Cloud Run service runs min-instances=1, but a
// just-deployed revision or a scale-up instance (1→max) can take tens of
// seconds to serve its first request while the ADK app finishes init, so this
// is generous; callers retry around it and keep boot in the "connecting" state
// meanwhile.
const AGENT_CARD_TIMEOUT_MS = 20_000;
const AGENT_CARD_MAX_ATTEMPTS = 5;
const AGENT_CARD_RETRY_BASE_MS = 1_500;

function abortError(): DOMException {
  return new DOMException("aborted", "AbortError");
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(abortError());
      },
      { once: true },
    );
  });
}

export async function fetchAgentCard(opts?: {
  timeoutMs?: number;
  signal?: AbortSignal;
}): Promise<AgentCard> {
  // Hard cap the lookup so a backend hang doesn't freeze boot indefinitely
  // (the chat input is disabled until the client resolves).
  const timeoutMs = opts?.timeoutMs ?? AGENT_CARD_TIMEOUT_MS;
  const controller = new AbortController();
  const onExternalAbort = () => controller.abort();
  opts?.signal?.addEventListener("abort", onExternalAbort, { once: true });
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`/.well-known/agent-card.json`, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error(
        `failed to fetch agent card: ${res.status} ${res.statusText}`,
      );
    }
    return (await res.json()) as AgentCard;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      // Distinguish a caller-requested cancel from a per-attempt timeout.
      if (opts?.signal?.aborted) throw err;
      throw new Error(`agent card fetch timed out after ${timeoutMs}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timeout);
    opts?.signal?.removeEventListener("abort", onExternalAbort);
  }
}

// Keep retrying with linear backoff while a new or scaling-up instance warms
// up, so boot shows a "connecting" spinner instead of flipping to
// "disconnected" on the first slow response. Stops immediately if the caller
// aborts.
async function fetchAgentCardWithRetry(
  signal?: AbortSignal,
): Promise<AgentCard> {
  let lastErr: unknown;
  for (let attempt = 1; attempt <= AGENT_CARD_MAX_ATTEMPTS; attempt++) {
    if (signal?.aborted) throw abortError();
    try {
      return await fetchAgentCard({ signal });
    } catch (err) {
      if (signal?.aborted) throw err;
      lastErr = err;
      if (attempt < AGENT_CARD_MAX_ATTEMPTS) {
        await sleep(AGENT_CARD_RETRY_BASE_MS * attempt, signal);
      }
    }
  }
  throw lastErr;
}

// Assemble the outgoing message parts. An empty text part (`{kind:"text",
// text:""}`) — which happens when a file is sent with no caption — can be
// rejected by strict model backends, so only include the text part when it
// carries content.
export function buildMessageParts(text: string, extraParts?: Part[]): Part[] {
  const parts: Part[] = [];
  if (text.trim()) parts.push({ kind: "text", text });
  if (extraParts?.length) parts.push(...extraParts);
  return parts;
}

export async function createLhaClient(
  opts: CreateLhaClientOptions,
): Promise<HorizonClient> {
  const card = await fetchAgentCardWithRetry(opts.signal);
  // Force the RPC URL to a relative same-origin path. The agent card built by
  // ADK declares an absolute URL based on APP_URL, which may not match the
  // browser's current origin (and in dev, the rewrites() block proxies /a2a to
  // the backend on :8001). Always hit /a2a same-origin.
  const proxiedCard: AgentCard = {
    ...card,
    url: RPC_PATH,
  };

  const { contextId } = opts;

  // Build a per-call A2AClient whose fetchImpl closes over THIS call's signal.
  // Previously a module-scoped `currentSignal` was clobbered when a
  // resubscribe overlapped with send() — the second assignment overwrote the
  // first, so the first stream could never be aborted.
  function buildClient(signal: AbortSignal | undefined): A2AClient {
    const fetchImpl: typeof fetch = (input, init) =>
      fetch(input, { ...init, signal: signal ?? init?.signal });
    return new A2AClient(proxiedCard, { fetchImpl });
  }

  // One shared instance for callers that read its fields; per-call instances
  // for anything that needs an abort signal.
  const client = buildClient(undefined);

  async function* sendStream(
    text: string,
    opts: SendOptions = {},
  ): AsyncGenerator<unknown, void, void> {
    const parts = buildMessageParts(text, opts.extraParts);
    const message: Message = {
      kind: "message",
      role: "user",
      messageId: opts.messageId ?? uuid(),
      contextId,
      parts,
    };
    const params: MessageSendParams = { message };
    const callClient = buildClient(opts.signal);
    for await (const event of callClient.sendMessageStream(params)) {
      yield event;
    }
  }

  async function* sendConfirmation(
    opts: SendConfirmationOptions,
  ): AsyncGenerator<unknown, void, void> {
    const message: Message = {
      kind: "message",
      role: "user",
      messageId: opts.messageId ?? uuid(),
      contextId,
      parts: [buildConfirmationDataPart(opts)],
    };
    const params: MessageSendParams = { message };
    const callClient = buildClient(opts.signal);
    for await (const event of callClient.sendMessageStream(params)) {
      yield event;
    }
  }

  async function* sendConfirmations(
    items: ConfirmationItem[],
    opts: { signal?: AbortSignal; messageId?: string } = {},
  ): AsyncGenerator<unknown, void, void> {
    const message: Message = {
      kind: "message",
      role: "user",
      messageId: opts.messageId ?? uuid(),
      contextId,
      parts: items.map(buildConfirmationDataPart),
    };
    const params: MessageSendParams = { message };
    const callClient = buildClient(opts.signal);
    for await (const event of callClient.sendMessageStream(params)) {
      yield event;
    }
  }

  async function getTask(
    taskId: string,
    opts: { signal?: AbortSignal } = {},
  ): Promise<Task> {
    const callClient = buildClient(opts.signal);
    // The new A2AClient returns a JSON-RPC envelope; unwrap to the Task or
    // surface the error so callers can handle it (e.g. task evicted from store).
    const response = await callClient.getTask({ id: taskId });
    if ("error" in response && response.error) {
      throw new Error(`tasks/get ${taskId}: ${formatRpcError(response.error)}`);
    }
    if (!("result" in response)) {
      throw new Error(`tasks/get ${taskId}: unexpected response shape`);
    }
    return response.result as Task;
  }

  // Real A2A tasks/cancel. cancelTask returns the Task directly (not an RPC
  // envelope like getTask), so there's nothing to unwrap — let errors throw and
  // leave it to the caller to ignore a benign "already terminal" rejection.
  async function cancelTask(
    taskId: string,
    opts: { signal?: AbortSignal } = {},
  ): Promise<void> {
    const callClient = buildClient(opts.signal);
    await callClient.cancelTask({ id: taskId });
  }

  async function* resubscribeTask(
    taskId: string,
    opts: { signal?: AbortSignal } = {},
  ): AsyncGenerator<unknown, void, void> {
    const callClient = buildClient(opts.signal);
    for await (const event of callClient.resubscribeTask({ id: taskId })) {
      yield event;
    }
  }

  return {
    card,
    client,
    sendStream,
    sendConfirmation,
    sendConfirmations,
    getTask,
    cancelTask,
    resubscribeTask,
    contextId,
  };
}
