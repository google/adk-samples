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

export type SandboxStatus =
  | { status: "idle"; user_id?: string }
  | { status: "provisioning"; started_at?: string; user_id?: string }
  | { status: "ready"; ready_at?: string; user_id?: string }
  | { status: "error"; error?: string; user_id?: string };

const KEY = "/lha/sandbox/status";

const fetcher = async (): Promise<SandboxStatus> => {
  const r = await fetch(KEY, { cache: "no-store" });
  if (!r.ok) throw new Error(`horizon-sandbox-status ${r.status}`);
  return r.json() as Promise<SandboxStatus>;
};

export function useSandboxStatus(): {
  data: SandboxStatus | undefined;
  error: Error | undefined;
  isLoading: boolean;
  refresh: () => Promise<unknown>;
} {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: qk.sandboxStatus(),
    queryFn: fetcher,
    // Sandbox status doesn't change without the user kicking warm; tab-focus
    // events shouldn't fire a fresh poll, especially during active streams
    // where they'd compete with the live request for the same connection pool.
    refetchOnWindowFocus: false,
    staleTime: 2_000,
    refetchInterval: (query) =>
      query.state.data?.status === "provisioning" ? 500 : false,
  });
  return {
    data: q.data,
    error: q.error ?? undefined,
    isLoading: q.isLoading,
    refresh: () => qc.invalidateQueries({ queryKey: qk.sandboxStatus() }),
  };
}

export async function warmSandbox(): Promise<void> {
  const r = await fetch("/lha/sandbox/warm", { method: "POST" });
  if (!r.ok) throw new Error(`warm-sandbox ${r.status}`);
}
