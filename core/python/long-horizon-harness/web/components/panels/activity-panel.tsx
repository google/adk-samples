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


import { memo, useMemo } from "react";
import { Wrench, CheckCircle2, XCircle } from "lucide-react";
import {
  useLhaState,
  type HorizonToolCall,
  type HorizonStateResponse,
} from "@/lib/horizon-state";
import { useNow } from "@/lib/now-context";
import { cn } from "@/lib/utils";

interface ActivityPanelProps {
  contextId: string | null;
}

function timeAgo(ts: number, now: number): string {
  const d = Math.max(0, now - ts);
  if (d < 1500) return "now";
  if (d < 60_000) return `${Math.floor(d / 1000)}s`;
  if (d < 3_600_000) return `${Math.floor(d / 60_000)}m`;
  return `${Math.floor(d / 3_600_000)}h`;
}

const selectToolCallLog = (d: HorizonStateResponse) => d.state.tool_call_log ?? [];

export function ActivityPanel({ contextId }: ActivityPanelProps) {
  const { data: callsRaw } = useLhaState(contextId, { select: selectToolCallLog });
  const now = useNow();
  const calls = callsRaw ?? [];
  const totalOk = calls.filter((c) => c.ok).length;
  const recentFailures = useMemo(
    () => calls.slice(-RECENT_FAILURE_WINDOW).filter((c) => !c.ok).length,
    [calls],
  );
  // Cap rendered rows so a 500-call session doesn't paint the whole list.
  const visible = useMemo(
    () => [...calls].reverse().slice(0, ACTIVITY_MAX_ROWS),
    [calls],
  );
  const hidden = Math.max(0, calls.length - visible.length);

  return (
    <div className="space-y-1 px-1">
      {calls.length > 0 && (
        <div className="flex items-center justify-end gap-2 pb-0.5 font-mono text-[10px] text-muted-foreground">
          {recentFailures > 0 && (
            <span
              className="rounded-full border border-destructive/40 bg-destructive/10 px-1.5 py-0 text-destructive"
              title={`${recentFailures} failed in the last ${RECENT_FAILURE_WINDOW} tool calls`}
            >
              {recentFailures} recent fail{recentFailures === 1 ? "" : "s"}
            </span>
          )}
          <span>
            {totalOk}/{calls.length} ok
          </span>
        </div>
      )}
      {calls.length === 0 ? (
        <div className="rounded-md border border-dashed bg-card/30 px-2 py-3 text-[11px] text-muted-foreground">
          {contextId
            ? "Tool calls will stream here as the agent works."
            : "Connecting…"}
        </div>
      ) : (
        <>
          {visible.map((c) => (
            <ToolRow key={`${c.ts}-${c.name}`} call={c} now={now} />
          ))}
          {hidden > 0 && (
            <div className="px-1 pt-1 text-[10px] italic text-muted-foreground/70">
              … {hidden} earlier tool call{hidden === 1 ? "" : "s"} hidden
            </div>
          )}
        </>
      )}
    </div>
  );
}

const ACTIVITY_MAX_ROWS = 50;
const RECENT_FAILURE_WINDOW = 10;

const ToolRow = memo(function ToolRow({ call, now }: { call: HorizonToolCall; now: number }) {
  const Icon = call.ok ? CheckCircle2 : XCircle;
  return (
    <div className="rounded-md border bg-card/50 px-2 py-1.5 text-[11px] motion-safe:animate-fade-in-up">
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <Wrench className="h-3 w-3 shrink-0 text-primary/70" />
          <span className="truncate font-mono text-foreground/90">{call.name}</span>
        </div>
        <div className="flex shrink-0 items-center gap-1.5 font-mono text-[10px] text-muted-foreground">
          <span>{timeAgo(call.ts, now)}</span>
          <Icon
            className={cn(
              "h-3 w-3",
              call.ok ? "text-emerald-500" : "text-destructive",
            )}
          />
        </div>
      </div>
      {call.args && (
        <div className="mt-0.5 truncate font-mono text-[10px] text-muted-foreground">
          {call.args}
        </div>
      )}
    </div>
  );
});
