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

export interface HorizonMemoryEntry {
  scope: "user" | "agent" | "user_profile";
  content: string;
  ts_ms: number;
  session_id: string | null;
}

export interface HorizonMemoriesResponse {
  entries: HorizonMemoryEntry[];
  counts: { user: number; agent: number; user_profile: number };
  has_more: boolean;
}

const fetchMemories = async (limit: number): Promise<HorizonMemoriesResponse> => {
  const r = await fetch(`/lha/memories?limit=${limit}&offset=0`, {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`horizon-memories ${r.status}`);
  return (await r.json()) as HorizonMemoriesResponse;
};

export interface UseLhaMemoriesResult {
  data: HorizonMemoriesResponse | undefined;
  error: Error | undefined;
  isLoading: boolean;
  refresh: () => Promise<unknown>;
}

export function useLhaMemories(limit: number): UseLhaMemoriesResult {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: qk.memories(limit),
    queryFn: () => fetchMemories(limit),
    refetchOnWindowFocus: true,
    staleTime: 10_000,
  });
  return {
    data: q.data,
    error: q.error ?? undefined,
    isLoading: q.isLoading,
    refresh: () => qc.invalidateQueries({ queryKey: qk.memories(limit) }),
  };
}
