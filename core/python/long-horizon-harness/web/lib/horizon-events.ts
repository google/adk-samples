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

// Helpers to extract user-visible content from A2A stream events.
//
// An A2A event is one of: Message (final reply), Task (in-progress),
// TaskStatusUpdateEvent, TaskArtifactUpdateEvent. Each carries Parts:
//   - TextPart       { kind: "text", text: string }
//   - FilePart       { kind: "file", file: ... }
//   - DataPart       { kind: "data", data: object, metadata?: {...} }

export interface ExtractedText {
  kind: "text";
  text: string;
}

export interface ExtractedData {
  kind: "data";
  mime: string | undefined;
  data: unknown;
}

export interface ExtractedTool {
  kind: "tool";
  callId: string | null;
  name: string;
  args: Record<string, unknown> | null;
  ok: boolean | null;
}

// ADK emits function_response after a tool completes. We pair it with the
// matching function_call (by callId) in chat-segments so the tool row can
// reveal its output when expanded.
export interface ExtractedToolResult {
  kind: "tool_result";
  callId: string | null;
  name: string;
  result: unknown;
}

// The clarify tool (horizon/tools/clarify.py) and ADK's framework-synthesized
// `adk_request_confirmation` carrying a clarify originalCall both collapse
// to this shape: a question, an optional finite set of choices, and the
// function_call_id needed for the continuation message.
export interface ExtractedClarify {
  kind: "clarify";
  callId: string | null;
  question: string;
  choices: string[];
}

// ADK's framework-synthesized confirmation for any non-clarify tool the agent
// wants to run with user approval (e.g. terminal). The hint is the prompt;
// originalCall lets the UI describe what's about to happen.
export interface ExtractedConfirmation {
  kind: "confirmation";
  callId: string | null;
  hint: string;
  originalCall: { name: string; args: Record<string, unknown> | null } | null;
  payload: Record<string, unknown> | null;
}

// The user's approve/decline/answer echo for a clarify/confirmation prompt,
// replayed from history as the adk_request_confirmation function_response. It
// carries the resolved ToolConfirmation so the matching card can re-render in
// its answered state instead of showing live (and now-stale) controls.
export interface ExtractedHitlResult {
  kind: "hitl_result";
  callId: string | null;
  confirmed: boolean;
  text: string | null;
}

export interface ExtractedFile {
  kind: "file";
  // A2A FilePart carries either inline base64 bytes or a URI; mimeType and
  // name are optional. We normalize to whichever the part provided.
  file: {
    bytes?: string;
    uri?: string;
    mimeType?: string;
    name?: string;
  };
}

export type Extracted =
  | ExtractedText
  | ExtractedData
  | ExtractedTool
  | ExtractedToolResult
  | ExtractedClarify
  | ExtractedConfirmation
  | ExtractedHitlResult
  | ExtractedFile;

export type Part = {
  kind?: string;
  text?: string;
  data?: unknown;
  file?: {
    bytes?: string;
    uri?: string;
    mimeType?: string;
    name?: string;
  };
  metadata?: {
    mimeType?: string;
    adk_type?: string;
    lha_ge_label?: boolean;
  } & Record<string, unknown>;
};

// ADK's A2A event converter exposes the model's intermediate function-call and
// function-response data through DataParts with metadata.adk_type. We surface
// both: function_call as the tool row header, function_response as the output
// payload shown when the user expands the row.

// Special tool names that get type-specific UI instead of the generic ToolChip.
const CLARIFY_TOOL_NAME = "clarify";
const CONFIRMATION_TOOL_NAME = "adk_request_confirmation";

// Mirrors horizon/a2a/executor.py _GE_TOOL_NAME_PREFIX: the backend prefixes flat
// tool names so GE's <agent>__<action> timeline filter renders them. Strip it
// back so the web shows flat labels and the special-case selectors still match.
const GE_TOOL_NAME_PREFIX = "horizon__";

function _stripGePrefix(name: string | undefined): string | undefined {
  return name?.startsWith(GE_TOOL_NAME_PREFIX)
    ? name.slice(GE_TOOL_NAME_PREFIX.length)
    : name;
}

// Mirrors horizon/a2a/executor.py _GE_LABEL_KEY: marks a text part the web
// should skip because it renders the same thing richly — the 📎 saved-artifact
// link (the web shows the artifact inline from its FilePart, so the text link
// would be a duplicate). GE has no inline file rendering and keeps the link.
const GE_LABEL_KEY = "lha_ge_label";

// Coerces an unknown shape into a finite string[] for clarify choices. Tolerant
// of missing/empty input — the renderer falls back to a free-form textarea.
function _toChoices(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter((x): x is string => typeof x === "string");
}

function _toRecord(raw: unknown): Record<string, unknown> | null {
  if (!raw || typeof raw !== "object") return null;
  return raw as Record<string, unknown>;
}

// ADK serializes a resolved ToolConfirmation either directly ({confirmed,
// payload, hint}) or wrapped by the client as {response: jsonString}. Unwrap
// both into the {confirmed, text} the resolved card needs. `text` is the chosen
// choice / typed answer (null for a bare approve/decline).
function _parseHitlResponse(
  response: unknown,
): { confirmed: boolean; text: string | null } | null {
  let rec = _toRecord(response);
  if (!rec) return null;
  if (typeof rec.response === "string" && Object.keys(rec).length === 1) {
    try {
      rec = _toRecord(JSON.parse(rec.response));
    } catch {
      return null;
    }
    if (!rec) return null;
  }
  if (typeof rec.confirmed !== "boolean") return null;
  const payload = _toRecord(rec.payload);
  const raw =
    (typeof payload?.choice === "string" && payload.choice) ||
    (typeof payload?.answer === "string" && payload.answer) ||
    null;
  return { confirmed: rec.confirmed, text: raw };
}

function _mapClarifyArgs(
  callId: string | null,
  args: Record<string, unknown> | null,
): ExtractedClarify | null {
  const question = typeof args?.question === "string" ? args.question : null;
  if (!question) return null;
  return {
    kind: "clarify",
    callId,
    question,
    choices: _toChoices(args?.choices),
  };
}

function _mapAdkConfirmation(
  callId: string | null,
  args: Record<string, unknown> | null,
): ExtractedClarify | ExtractedConfirmation | null {
  const original = _toRecord(args?.originalFunctionCall);
  const confirmation = _toRecord(args?.toolConfirmation);
  // Confirmations wrapping a clarify call collapse to the clarify UI — the
  // user picks a choice and we send the continuation. Keep the OUTER callId
  // because that's the function_call the agent is awaiting a response for.
  if (original?.name === CLARIFY_TOOL_NAME) {
    const innerArgs = _toRecord(original.args);
    const fromArgs = _mapClarifyArgs(callId, innerArgs);
    if (fromArgs) return fromArgs;
    // Some flows pass question/choices on the confirmation payload, not on
    // the inner args. Fall back to that shape.
    const payload = _toRecord(confirmation?.payload);
    return _mapClarifyArgs(callId, payload);
  }
  const hint = typeof confirmation?.hint === "string" ? confirmation.hint : "";
  return {
    kind: "confirmation",
    callId,
    hint,
    originalCall: original
      ? {
          name: typeof original.name === "string" ? original.name : "tool",
          args: _toRecord(original.args),
        }
      : null,
    payload: _toRecord(confirmation?.payload),
  };
}

function partsOf(event: unknown): Part[] {
  if (!event || typeof event !== "object") return [];
  const e = event as Record<string, unknown>;

  // Message event — { kind: "message", role, parts: [...] }
  // Skip if the message is the user's input echoed back (role=user).
  if (e.kind === "message" && Array.isArray(e.parts)) {
    if (e.role === "user") return [];
    return e.parts as Part[];
  }

  // Task event — { kind: "task", status: {...}, artifacts?: [{parts: [...]}, ...], history?: [{parts: [...]}, ...] }
  if (e.kind === "task") {
    const out: Part[] = [];
    const artifacts = e.artifacts as Array<{ parts?: Part[] }> | undefined;
    artifacts?.forEach((a) => a.parts && out.push(...a.parts));
    return out;
  }

  // TaskStatusUpdateEvent — { kind: "status-update", status: { message: { role, parts: [...] }, state, ... } }
  // The first event echoes the user's input as { state: "submitted", message: { role: "user", ... } }.
  // Skip user-authored messages so they don't get merged into the assistant bubble.
  if (e.kind === "status-update") {
    const status = e.status as
      | { message?: { role?: string; parts?: Part[] } }
      | undefined;
    if (status?.message?.role === "user") return [];
    return status?.message?.parts ?? [];
  }

  // TaskArtifactUpdateEvent — on the new-impl converter path the model's
  // streamed text and files both arrive here (append=true deltas into one
  // artifact; the redundant final aggregate is stripped server-side). Tool
  // calls are surfaced onto status-update messages, so artifact-updates carry
  // text/files only — return all parts.
  if (e.kind === "artifact-update") {
    const artifact = e.artifact as { parts?: Part[] } | undefined;
    return artifact?.parts ?? [];
  }
  return [];
}

// Per-part conversion shared between live-stream extraction and Task
// history replay (parts come from tasks/get instead of the SDK stream).
// Exported so replay can call it directly without going through partsOf
// (which deduplicates user-echo events that replay needs to keep).
export function mapPart(p: Part): Extracted | null {
  if (p.kind === "text" && typeof p.text === "string") {
    // GE-only text (the 📎 saved-artifact link) — the web renders the artifact
    // inline from its FilePart, so skip the marked link to avoid a duplicate.
    if (p.metadata?.[GE_LABEL_KEY] === true) return null;
    return { kind: "text", text: p.text };
  }
  if (p.kind === "file" && p.file && (p.file.bytes || p.file.uri)) {
    return {
      kind: "file",
      file: {
        bytes: p.file.bytes,
        uri: p.file.uri,
        mimeType: p.file.mimeType,
        name: p.file.name,
      },
    };
  }
  if (p.kind === "data") {
    const adkType = p.metadata?.adk_type;
    if (adkType === "function_call") {
      const d = (p.data ?? {}) as {
        id?: string;
        name?: string;
        args?: Record<string, unknown>;
      };
      const callId = typeof d.id === "string" ? d.id : null;
      const args = d.args ?? null;
      const name = _stripGePrefix(d.name);
      if (name === CLARIFY_TOOL_NAME) {
        // Suppress the model's raw clarify function_call. ADK pairs it with a
        // synthesized adk_request_confirmation that gets a fresh callId and is
        // the only id ADK matches the user's answer against — that event is
        // what renders the interactive card. Surfacing this one too produced a
        // duplicate card whose callId couldn't resume the agent.
        if (_mapClarifyArgs(callId, args)) return null;
        // Malformed clarify (no question): fall through to a generic tool row.
      }
      if (name === CONFIRMATION_TOOL_NAME) {
        const confirm = _mapAdkConfirmation(callId, args);
        if (confirm) return confirm;
      }
      return {
        kind: "tool",
        callId,
        name: name ?? "tool",
        args,
        ok: null,
      };
    }
    if (adkType === "function_response") {
      const d = (p.data ?? {}) as {
        id?: string;
        name?: string;
        response?: unknown;
      };
      const name = _stripGePrefix(d.name);
      // The adk_request_confirmation response is the user's approve/decline/
      // answer echo. It never joins the visible tool log; instead it resolves
      // the matching card to its answered state (live + replay).
      if (name === CONFIRMATION_TOOL_NAME) {
        const resolved = _parseHitlResponse(d.response);
        if (!resolved) return null;
        return {
          kind: "hitl_result",
          callId: typeof d.id === "string" ? d.id : null,
          confirmed: resolved.confirmed,
          text: resolved.text,
        };
      }
      return {
        kind: "tool_result",
        callId: typeof d.id === "string" ? d.id : null,
        name: typeof name === "string" ? name : "tool",
        result: d.response,
      };
    }
    const mime = p.metadata?.mimeType;
    return { kind: "data", mime, data: p.data };
  }
  return null;
}

export function extractParts(event: unknown): Extracted[] {
  return partsOf(event)
    .map(mapPart)
    .filter((x): x is Extracted => x !== null);
}

export function isFinal(event: unknown): boolean {
  if (!event || typeof event !== "object") return false;
  const e = event as Record<string, unknown>;
  if (e.kind === "message") return true;
  if (e.kind === "status-update" && e.final === true) return true;
  // Task with terminal state
  if (e.kind === "task") {
    const status = e.status as { state?: string } | undefined;
    const state = status?.state;
    return state === "completed" || state === "failed" || state === "canceled" || state === "rejected";
  }
  return false;
}

// The A2A task id an event belongs to. Status-update and message events carry
// `taskId`; a Task event carries it as `id`. Used to record which tasks the
// live stream drove so resubscribe never re-attaches to one of them.
export function extractTaskId(event: unknown): string | null {
  if (!event || typeof event !== "object") return null;
  const e = event as Record<string, unknown>;
  if (typeof e.taskId === "string") return e.taskId;
  if (e.kind === "task" && typeof e.id === "string") return e.id;
  return null;
}
