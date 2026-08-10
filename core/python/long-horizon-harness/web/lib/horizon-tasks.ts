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


import { useCallback } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { qk } from "./query-keys";

export interface HorizonTaskSummary {
  id: string;
  status:
    | "submitted"
    | "working"
    | "input-required"
    | "completed"
    | "canceled"
    | "failed"
    | "rejected"
    | "auth-required"
    | "unknown";
  isActive: boolean;
}

const TERMINAL_STATES: ReadonlySet<HorizonTaskSummary["status"]> = new Set([
  "completed",
  "canceled",
  "failed",
  "rejected",
]);

export function isTerminalTaskStatus(status: HorizonTaskSummary["status"]): boolean {
  return TERMINAL_STATES.has(status);
}

export const fetcher = async <T>(url: string): Promise<T> => {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`horizon-tasks ${r.status}`);
  return r.json() as Promise<T>;
};

export function tasksKey(contextId: string | null): string | null {
  if (!contextId) return null;
  return `/lha/contexts/${encodeURIComponent(contextId)}/tasks`;
}

export interface UseLhaTasksResult {
  tasks: HorizonTaskSummary[] | undefined;
  activeTaskId: string | null;
  error: Error | undefined;
  isLoading: boolean;
  refresh: () => Promise<unknown>;
}

export function useLhaTasks(
  contextId: string | null,
): UseLhaTasksResult {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: qk.tasks(contextId ?? ""),
    queryFn: () => fetcher<HorizonTaskSummary[]>(tasksKey(contextId)!),
    enabled: !!contextId,
    refetchInterval: 10_000,
    refetchOnWindowFocus: true,
    staleTime: 2_000,
    select: (tasks) => ({
      tasks,
      // A stale pointer over a settled task would lock the composer forever.
      activeTaskId:
        tasks.find((t) => t.isActive && !isTerminalTaskStatus(t.status))?.id ??
        null,
    }),
  });
  const refresh = useCallback(
    () => qc.invalidateQueries({ queryKey: qk.tasks(contextId ?? "") }),
    [qc, contextId],
  );
  return {
    tasks: q.data?.tasks,
    activeTaskId: q.data?.activeTaskId ?? null,
    error: q.error ?? undefined,
    isLoading: q.isLoading,
    refresh,
  };
}
