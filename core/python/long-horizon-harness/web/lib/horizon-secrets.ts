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

export interface HorizonSecretEntry {
  name: string;
  created_at: number | null;
}

export interface HorizonSecretsResponse {
  secrets: HorizonSecretEntry[];
}

const KEY = "/lha/secrets";

const fetcher = async (): Promise<HorizonSecretsResponse> => {
  const r = await fetch(KEY, { cache: "no-store" });
  if (!r.ok) throw new Error(`horizon-secrets ${r.status}`);
  return (await r.json()) as HorizonSecretsResponse;
};

export interface UseLhaSecretsResult {
  data: HorizonSecretsResponse | undefined;
  error: Error | undefined;
  isLoading: boolean;
  refresh: () => Promise<unknown>;
}

export function useLhaSecrets(): UseLhaSecretsResult {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: qk.secrets(),
    queryFn: fetcher,
    refetchOnWindowFocus: true,
    staleTime: 10_000,
  });
  return {
    data: q.data,
    error: q.error ?? undefined,
    isLoading: q.isLoading,
    refresh: () => qc.invalidateQueries({ queryKey: qk.secrets() }),
  };
}

export async function putSecret(name: string, value: string): Promise<void> {
  const r = await fetch(`/lha/secrets/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
  if (!r.ok) throw new Error(`putSecret ${r.status}`);
}

export async function deleteSecret(name: string): Promise<void> {
  const r = await fetch(`/lha/secrets/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`deleteSecret ${r.status}`);
}

export interface ImportDotenvResult {
  valid: string[];
  skipped: { lineno: number; reason: string }[];
  conflicts: string[];
  committed: boolean;
}

export async function importDotenv(
  content: string,
  commit: boolean,
): Promise<ImportDotenvResult> {
  const r = await fetch("/lha/secrets/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, commit }),
  });
  if (!r.ok) throw new Error(`importDotenv ${r.status}`);
  return (await r.json()) as ImportDotenvResult;
}
