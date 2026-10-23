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


import { useState } from "react";
import { Cloud, Loader2 } from "lucide-react";
import {
  useGoogleConnection,
  connectGoogle,
  disconnectGoogle,
  isExpired,
} from "@/lib/gcp-connection";

function expiresLabel(expiresAt: number | null | undefined): string {
  if (!expiresAt) return "";
  const mins = Math.max(0, Math.round((expiresAt * 1000 - Date.now()) / 60000));
  return ` · expires in ~${mins}m`;
}

export function GcpConnectCard() {
  const { status, refresh } = useGoogleConnection();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const gcp = status?.gcp;

  async function run(fn: () => Promise<void>) {
    setErr(null);
    setBusy(true);
    try {
      await fn();
      for (const d of [3000, 6000, 10000]) setTimeout(() => void refresh(), d);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-2 rounded border border-border p-2 text-xs">
      <div className="flex items-center gap-2 font-medium text-foreground">
        <Cloud className="h-4 w-4 text-primary" />
        Google Cloud
      </div>

      {gcp?.connected ? (
        <>
          <div className={isExpired(gcp.expires_at) ? "text-destructive" : "text-muted-foreground"}>
            {isExpired(gcp.expires_at)
              ? "Connection expired — reconnect"
              : `Connected${expiresLabel(gcp.expires_at)}`}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => run(() => connectGoogle("gcp"))}
              disabled={busy}
              className="rounded border border-border px-2 py-1 hover:bg-muted disabled:opacity-50"
            >
              Reconnect
            </button>
            <button
              onClick={() => run(() => disconnectGoogle("gcp").then(refresh))}
              disabled={busy}
              className="rounded border border-border px-2 py-1 hover:bg-muted disabled:opacity-50"
            >
              Disconnect
            </button>
          </div>
        </>
      ) : (
        <button
          onClick={() => run(() => connectGoogle("gcp"))}
          disabled={busy}
          className="inline-flex w-fit items-center gap-1.5 rounded bg-primary px-2.5 py-1 font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
          Connect Google Cloud
        </button>
      )}

      <div className="text-[10px] text-muted-foreground">
        Short-lived token (~1h) for gcloud/bq with full cloud-platform access;
        reconnect when it lapses.
      </div>
      {err ? <div className="text-destructive">{err}</div> : null}
    </div>
  );
}
