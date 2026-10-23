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


import { useMemo } from "react";
import { CheckCircle2, XCircle, Clock } from "lucide-react";
import {
  useLhaState,
  type HorizonDelegateRun,
  type HorizonStateResponse,
} from "@/lib/horizon-state";
import { cn } from "@/lib/utils";

interface DelegationTreeProps {
  contextId: string | null;
}

const STATUS_META: Record<
  string,
  { Icon: typeof CheckCircle2; tint: string }
> = {
  completed: { Icon: CheckCircle2, tint: "text-emerald-500" },
  timeout: { Icon: Clock, tint: "text-amber-500" },
  halted: { Icon: XCircle, tint: "text-destructive" },
};

function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  return `${m}m${s}s`;
}

const selectDelegateSlice = (d: HorizonStateResponse) => ({
  runs: d.state.delegate_history ?? [],
});

export function DelegationTree({ contextId }: DelegationTreeProps) {
  const { data } = useLhaState(contextId, { select: selectDelegateSlice });
  const runs = data?.runs ?? [];
  const reversedRuns = useMemo(() => [...runs].reverse(), [runs]);

  return (
    <div className="space-y-3 px-1">
      <div className="rounded-md border bg-card/50 px-2 py-2 text-[11px]">
        <div className="flex items-center gap-1.5 font-mono text-primary">
          <span className="h-1.5 w-1.5 rounded-full bg-primary motion-safe:animate-pulse" />
          root_agent
        </div>
        <div className="mt-1 ml-3 text-muted-foreground">
          {contextId ? "Top-level agent for this session" : "Connecting…"}
        </div>
      </div>

      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
          Delegated runs
        </div>
        {runs.length === 0 ? (
          <div className="rounded-md border border-dashed bg-card/30 px-2 py-3 text-[11px] text-muted-foreground">
            When the agent hands part of a task off to a helper agent, those
            runs show up here.
          </div>
        ) : (
          reversedRuns.map((r) => (
            <DelegateRow key={r.task_id ?? `${r.ts}-${r.goal}`} run={r} />
          ))
        )}
      </div>
    </div>
  );
}

function DelegateRow({ run }: { run: HorizonDelegateRun }) {
  const running = run.status === "running";
  const meta = STATUS_META[run.status] ?? STATUS_META.completed;
  const Icon = meta.Icon;
  return (
    <div className="rounded-md border bg-card/50 px-2 py-1.5 text-[11px] motion-safe:animate-fade-in-up">
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-1.5">
          {running ? (
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-primary motion-safe:animate-pulse" />
          ) : (
            <Icon className={cn("h-3 w-3 shrink-0", meta.tint)} />
          )}
          <span className="truncate font-mono text-foreground/90">{run.goal}</span>
        </div>
        <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
          {fmtMs(run.duration_ms)}
        </span>
      </div>
      {run.summary && (
        <div className="mt-1 line-clamp-2 text-muted-foreground">
          {run.summary}
        </div>
      )}
      <div className="mt-1 flex gap-2 font-mono text-[10px] text-muted-foreground">
        <span>{run.iterations} iter</span>
        <span>·</span>
        <span className={running ? "text-primary" : meta.tint}>{run.status}</span>
      </div>
    </div>
  );
}
