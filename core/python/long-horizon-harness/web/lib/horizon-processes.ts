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

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { qk } from "./query-keys";

export interface HorizonProcess {
  session_id: string;
  command: string;
  running: boolean;
  exit_code: number | null;
  idle_seconds: number;
  output_size: number;
  pid: number;
  started_at: number;
}

export interface HorizonProcessesResponse {
  processes: HorizonProcess[];
}

const KEY = "/lha/processes";

const fetcher = async (): Promise<HorizonProcessesResponse> => {
  const r = await fetch(KEY, { cache: "no-store" });
  if (!r.ok) throw new Error(`horizon-processes ${r.status}`);
  return (await r.json()) as HorizonProcessesResponse;
};

export function hasRunning(resp: HorizonProcessesResponse | undefined): boolean {
  return resp?.processes.some((p) => p.running) ?? false;
}

export interface UseProcessesResult {
  data: HorizonProcessesResponse | undefined;
  error: Error | undefined;
  isLoading: boolean;
  refresh: () => Promise<unknown>;
}

export function useProcesses(): UseProcessesResult {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: qk.processes(),
    queryFn: fetcher,
    refetchOnWindowFocus: true,
    // Hybrid liveness: the Section only mounts this hook while expanded, so
    // polling is scoped to "expanded AND >=1 running"; idle otherwise.
    refetchInterval: (query) => (hasRunning(query.state.data) ? 5000 : false),
    staleTime: 2_000,
  });
  return {
    data: q.data,
    error: q.error ?? undefined,
    isLoading: q.isLoading,
    refresh: () => qc.invalidateQueries({ queryKey: qk.processes() }),
  };
}

export async function killProcess(sessionId: string): Promise<void> {
  const r = await fetch(`/lha/processes/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`killProcess ${r.status}`);
}
