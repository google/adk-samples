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


import { AlertTriangle, Activity, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useLhaState, type HorizonStateResponse } from "@/lib/horizon-state";

interface GuardrailBannerProps {
  contextId: string | null;
  bootError: string | null;
}

const selectGuardrailSlice = (d: HorizonStateResponse) => ({
  halt: d.state.halt_reason ?? d.state.iteration_halt_reason ?? null,
  lastError: d.state.last_error ?? null,
  sandbox: d.state.sandbox_status ?? null,
  iteration: d.state.iteration,
  toolCalls: d.state.tool_calls_this_iteration,
});

export function GuardrailBanner({ contextId, bootError }: GuardrailBannerProps) {
  const { data } = useLhaState(contextId, { select: selectGuardrailSlice });
  const halt = data?.halt ?? null;
  const lastError = data?.lastError ?? null;
  const sandbox = data?.sandbox ?? null;

  if (bootError) {
    return (
      <div
        className={cn(
          "flex items-center gap-2 border-b border-destructive/40 bg-destructive/10 px-4 py-2 text-xs text-destructive",
        )}
      >
        <AlertTriangle className="h-3.5 w-3.5" />
        <span className="font-mono">disconnected</span>
        <span className="text-destructive/80">{bootError}</span>
      </div>
    );
  }
  if (sandbox?.status === "provisioning") {
    return (
      <div className="flex items-start gap-2 border-b border-[hsl(var(--lh-shuttle)/0.4)] bg-[hsl(var(--lh-shuttle)/0.1)] px-4 py-2 text-xs text-foreground motion-safe:animate-fade-in-up">
        <Loader2 className="mt-0.5 h-3.5 w-3.5 animate-spin" />
        <div className="flex flex-col">
          <span className="font-medium">Spinning up your sandbox…</span>
          <span className="text-muted-foreground">
            First turn takes ~10–30s to provision your isolated /workspace container.
          </span>
        </div>
      </div>
    );
  }
  if (sandbox?.status === "error") {
    return (
      <div className="flex items-start gap-2 border-b border-destructive/40 bg-destructive/10 px-4 py-2 text-xs text-destructive motion-safe:animate-fade-in-up">
        <AlertTriangle className="mt-0.5 h-3.5 w-3.5" />
        <div className="flex flex-col">
          <span className="font-medium">Sandbox provisioning failed</span>
          {sandbox.error ? (
            <span className="text-destructive/80">{sandbox.error}</span>
          ) : null}
        </div>
      </div>
    );
  }
  if (halt) {
    return (
      <div className="flex items-start gap-2 border-b border-destructive/40 bg-destructive/10 px-4 py-2 text-xs text-destructive motion-safe:animate-fade-in-up">
        <AlertTriangle className="mt-0.5 h-3.5 w-3.5" />
        <div className="flex flex-col">
          <span className="font-mono uppercase tracking-wide">guardrail · halt</span>
          <span className="text-destructive/90">{halt}</span>
          {lastError ? (
            <span className="mt-0.5 text-[10px] text-destructive/70">
              last error: {lastError}
            </span>
          ) : null}
        </div>
      </div>
    );
  }
  if (!contextId || !data?.iteration) return null;
  return (
    <div className="flex items-center gap-3 border-b bg-[hsl(var(--lh-shuttle)/0.08)] px-4 py-1.5 text-[11px] text-muted-foreground">
      <span className="flex items-center gap-1">
        <Activity className="h-3 w-3" />
        iter {data.iteration} · tool calls {data.toolCalls}
      </span>
    </div>
  );
}
