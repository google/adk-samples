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


import { AlertTriangle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { SandboxStatus } from "@/lib/horizon-sandbox-status";

export function SandboxWarmupBanner({
  status,
  warmError,
  className,
}: {
  status: SandboxStatus | undefined;
  // POST /lha/sandbox/warm failure surfaced from the boot effect. The status
  // endpoint won't report this — it only knows about warms it actually
  // received — so we need the client-side signal too.
  warmError?: string | null;
  className?: string;
}) {
  const banner = (message: string) => (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "mx-auto flex w-full max-w-2xl items-center gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-1.5 text-xs text-destructive",
        className,
      )}
    >
      <AlertTriangle className="h-3 w-3 shrink-0" />
      <span className="min-w-0 truncate">{message}</span>
    </div>
  );

  if (warmError) {
    return banner(`Couldn't kick off workspace warmup: ${warmError}. First message will retry.`);
  }
  if (!status) return null;
  if (status.status === "idle" || status.status === "ready") return null;

  if (status.status === "error") {
    return banner(
      `Workspace failed to prepare${status.error ? `: ${status.error}` : ""}. First message will retry.`,
    );
  }

  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "mx-auto flex w-full max-w-2xl items-center gap-2 rounded-md border border-border/40 bg-muted/30 px-3 py-1.5 text-xs text-muted-foreground",
        className,
      )}
    >
      <Loader2 className="h-3 w-3 shrink-0 animate-spin" />
      <span>Preparing your workspace…</span>
    </div>
  );
}
