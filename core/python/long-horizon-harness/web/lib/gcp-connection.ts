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

export interface GoogleConnectionStatus {
  gcp: { connected: boolean; expires_at: number | null };
  workspace: {
    connected: boolean;
    expires_at: number | null;
    surfaces: string[];
    readonly: boolean;
  };
  surfaces: string[]; // available Workspace surfaces
}

const fetcher = async (): Promise<GoogleConnectionStatus> => {
  const r = await fetch("/lha/gcp/status", { cache: "no-store" });
  if (!r.ok) throw new Error(`gcp-status ${r.status}`);
  return (await r.json()) as GoogleConnectionStatus;
};

export function useGoogleConnection() {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: qk.gcpConnection(),
    queryFn: fetcher,
    refetchOnWindowFocus: true,
    staleTime: 5_000,
  });
  return {
    status: q.data,
    isLoading: q.isLoading,
    refresh: () => qc.invalidateQueries({ queryKey: qk.gcpConnection() }),
  };
}

/** True when an access token's expiry (unix seconds) has passed — the tokens are
 * access-only (~1h, no refresh), so a lapsed one is no longer a live connection. */
export function isExpired(expiresAt: number | null | undefined): boolean {
  return expiresAt != null && expiresAt * 1000 <= Date.now();
}

export type ConnectTarget = "gcp" | "workspace";

const CONNECT_CAUTION =
  "Caution: this sandbox has access to the internet. Connecting lets the " +
  "agent use your Google access (~1h) from inside it. Continue?";

/** Start a consent flow for one target; opens Google's consent in a new window. */
export async function connectGoogle(
  target: ConnectTarget,
  opts: { surfaces?: string[]; readonly?: boolean } = {},
): Promise<void> {
  if (!window.confirm(CONNECT_CAUTION)) return;
  const params = new URLSearchParams({ target });
  if (target === "workspace") {
    params.set("readonly", String(opts.readonly ?? true));
    params.set("surfaces", (opts.surfaces ?? []).join(","));
  }
  const r = await fetch(`/lha/gcp/connect?${params.toString()}`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(`gcp-connect ${r.status}`);
  const { auth_url } = (await r.json()) as { auth_url: string };
  window.open(auth_url, "_blank", "noopener,noreferrer");
}

export async function disconnectGoogle(target: ConnectTarget): Promise<void> {
  const r = await fetch(`/lha/gcp/disconnect?target=${target}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`gcp-disconnect ${r.status}`);
}
