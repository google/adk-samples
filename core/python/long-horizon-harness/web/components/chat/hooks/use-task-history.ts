"use client";
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


import { useEffect, useState } from "react";
import type { Task } from "@a2a-js/sdk";
import type { HorizonClient } from "@/lib/a2a-client";
import {
  resolveHitlAcrossTasks,
  resolveToolResultsAcrossTasks,
  taskToChatMessages,
} from "@/lib/task-to-segments";
import { isTerminalTaskStatus, type HorizonTaskSummary } from "@/lib/horizon-tasks";
import { getTaskHistoryCache, type TaskCacheEntry } from "@/lib/task-history-cache";
import type { ChatMessage } from "../chat-shell";

export interface UseTaskHistoryArgs {
  contextId: string | null;
  tasks: HorizonTaskSummary[] | undefined;
  client: HorizonClient | null;
}

export interface UseTaskHistoryResult {
  historyMessages: ChatMessage[] | null;
  historyLoading: boolean;
}

// Stable signature so we only re-run when task identities or status actually
// change. isActive flips on every turn but doesn't affect history; excluding it
// avoids refetching every completed task on each transition.
function _tasksSignature(tasks: HorizonTaskSummary[] | undefined): string {
  if (!tasks) return "";
  return tasks.map((t) => `${t.id}:${t.status}`).join("|");
}

export function useTaskHistory({
  contextId,
  tasks,
  client,
}: UseTaskHistoryArgs): UseTaskHistoryResult {
  const [historyMessages, setHistoryMessages] = useState<ChatMessage[] | null>(
    null,
  );
  const [historyLoading, setHistoryLoading] = useState(false);
  const signature = _tasksSignature(tasks);

  useEffect(() => {
    if (tasks === undefined) {
      // tasks undefined = query still loading. Don't clobber a previous render
      // with null while a refresh is in flight.
      return;
    }
    if (!contextId || tasks.length === 0) {
      setHistoryMessages([]);
      return;
    }

    // Per-chat cache survives client swaps and component switches, keyed by
    // contextId. Only terminal tasks are stored (see below), so an in-flight
    // task is always re-fetched and re-subscribed.
    const entries = getTaskHistoryCache(contextId);

    // Evict cache entries no longer in `tasks` so they don't leak across edits.
    const liveIds = new Set(tasks.map((t) => t.id));
    for (const id of entries.keys()) {
      if (!liveIds.has(id)) entries.delete(id);
    }

    const rebuildFrom = (source: Map<string, TaskCacheEntry>): ChatMessage[] => {
      const rebuilt: ChatMessage[] = [];
      // The same messageId can appear in multiple tasks' history (ADK carries
      // the initiating user message into each follow-up task). Keep the first
      // occurrence so React keys stay unique and ordering is stable.
      const seenIds = new Set<string>();
      const orderedTasks: Task[] = [];
      for (let i = 0; i < tasks.length; i++) {
        const entry = source.get(tasks[i].id);
        if (!entry) continue;
        orderedTasks.push(entry.task);
        const msgs = taskToChatMessages({
          task: entry.task,
          taskIndex: i,
        });
        for (const msg of msgs) {
          if (seenIds.has(msg.id)) continue;
          seenIds.add(msg.id);
          rebuilt.push(msg);
        }
      }
      // A confirmation card and its approve/decline echo can straddle two
      // tasks (ADK resumes into a new task), so resolve HITL across the whole
      // assembled list — the per-task resolve inside taskToChatMessages can't.
      const withHitl = resolveHitlAcrossTasks(rebuilt, orderedTasks);
      // Likewise a HITL-gated tool's real function_response can land in a later
      // task than its spinning function_call; bridge it so the row stops
      // spinning on reload.
      return resolveToolResultsAcrossTasks(withHitl, orderedTasks);
    };

    // A task needs fetching if it isn't cached with a matching status. Because
    // we only cache terminal tasks, any non-terminal task is always here.
    const toFetch = tasks.filter((t) => entries.get(t.id)?.status !== t.status);

    // Fast path: every task is terminal and already cached. Rebuild from cache
    // synchronously — no getTask() calls, no loading flash, and no need to wait
    // for the A2A client to finish booting.
    if (toFetch.length === 0) {
      setHistoryMessages(rebuildFrom(entries));
      setHistoryLoading(false);
      return;
    }

    if (!client) {
      // Can't fetch the missing/changed tasks until the client boots. A cached
      // view (if any) is already showing from a prior render.
      return;
    }

    let cancelled = false;
    setHistoryLoading(true);

    (async () => {
      const fetched = await Promise.all(
        toFetch.map(async (s) => {
          try {
            const task = await client.getTask(s.id);
            return { id: s.id, status: s.status, task };
          } catch (err) {
            // Don't cache failures — a transient backend hiccup must not hide
            // the task forever. The next status flip will retry.
            console.warn(`useTaskHistory: skipping task ${s.id}`, err);
            return null;
          }
        }),
      );
      if (cancelled) return;
      for (const r of fetched) {
        if (!r) continue;
        // Only cache terminal tasks; in-flight ones must always be re-fetched
        // and re-subscribed so a running background turn is never served stale.
        if (isTerminalTaskStatus(r.status)) {
          entries.set(r.id, { status: r.status, task: r.task });
        } else {
          // Rebuild needs the task object even for an in-flight turn, but it
          // must not persist — drop any stale terminal entry under this id.
          entries.delete(r.id);
        }
      }

      // Overlay the just-fetched tasks (including the in-flight one we did not
      // persist) on top of the cached entries so they render this pass.
      const overlay = new Map(entries);
      for (const r of fetched) {
        if (r) overlay.set(r.id, { status: r.status, task: r.task });
      }
      setHistoryMessages(rebuildFrom(overlay));
      setHistoryLoading(false);
    })();

    return () => {
      cancelled = true;
    };
    // signature captures task changes; contextId/client capture chat + boot.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, contextId, client]);

  return { historyMessages, historyLoading };
}
