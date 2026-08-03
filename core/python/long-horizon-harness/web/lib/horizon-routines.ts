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

export interface HorizonRoutine {
  id: string;
  task: string;
  schedule: string;
  next_fire_at: string;
  secrets: string[];
  delivery: string;
}

export interface HorizonRoutinesResponse {
  routines: HorizonRoutine[];
}

const KEY = "/lha/routines";

const fetcher = async (): Promise<HorizonRoutinesResponse> => {
  const r = await fetch(KEY, { cache: "no-store" });
  if (!r.ok) throw new Error(`horizon-routines ${r.status}`);
  return (await r.json()) as HorizonRoutinesResponse;
};

export interface UseRoutinesResult {
  data: HorizonRoutinesResponse | undefined;
  error: Error | undefined;
  isLoading: boolean;
  refresh: () => Promise<unknown>;
}

export function useRoutines(): UseRoutinesResult {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: qk.routines(),
    queryFn: fetcher,
    refetchOnWindowFocus: true,
    staleTime: 10_000,
  });
  return {
    data: q.data,
    error: q.error ?? undefined,
    isLoading: q.isLoading,
    refresh: () => qc.invalidateQueries({ queryKey: qk.routines() }),
  };
}

export async function deleteRoutine(id: string): Promise<void> {
  const r = await fetch(`/lha/routines/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`deleteRoutine ${r.status}`);
}
