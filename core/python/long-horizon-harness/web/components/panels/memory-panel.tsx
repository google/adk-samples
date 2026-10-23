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


import { memo, useState } from "react";
import { User, Bot, IdCard } from "lucide-react";
import {
  useLhaMemories,
  type HorizonMemoryEntry,
} from "@/lib/horizon-memories";
import { useNow } from "@/lib/now-context";
import { cn } from "@/lib/utils";

interface MemoryPanelProps {
  contextId: string | null;
}

export const MEMORY_PANEL_PAGE_LIMIT = 5;
const PAGE_STEP = 5;

const SCOPE_META: Record<
  HorizonMemoryEntry["scope"],
  { label: string; Icon: typeof User; tint: string; hint: string }
> = {
  user: {
    label: "user",
    Icon: User,
    tint: "text-sky-500",
    hint: "Facts you told the agent across all your sessions",
  },
  agent: {
    label: "agent",
    Icon: Bot,
    tint: "text-primary",
    hint: "Things the agent decided to remember on its own",
  },
  user_profile: {
    label: "profile",
    Icon: IdCard,
    tint: "text-emerald-500",
    hint: "Long-term profile facts that persist across sessions",
  },
};

function timeAgo(ts: number, now: number): string {
  const d = Math.max(0, now - ts);
  if (d < 1500) return "just now";
  if (d < 60_000) return `${Math.floor(d / 1000)}s ago`;
  if (d < 3_600_000) return `${Math.floor(d / 60_000)}m ago`;
  if (d < 86_400_000) return `${Math.floor(d / 3_600_000)}h ago`;
  return new Date(ts).toLocaleDateString();
}

export function MemoryPanel({ contextId }: MemoryPanelProps) {
  const [limit, setLimit] = useState(MEMORY_PANEL_PAGE_LIMIT);
  const { data, isLoading } = useLhaMemories(limit);
  const now = useNow();

  const counts = data?.counts ?? { user: 0, agent: 0, user_profile: 0 };
  const entries = data?.entries ?? [];
  const hasMore = data?.has_more ?? false;
  const showEmptyState = !isLoading && data !== undefined && entries.length === 0;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-1.5 px-1">
        {(["user", "agent", "user_profile"] as const).map((scope) => {
          const meta = SCOPE_META[scope];
          const count = counts[scope] ?? 0;
          const Icon = meta.Icon;
          return (
            <div
              key={scope}
              className="rounded-md border bg-card/50 px-2 py-2"
              title={meta.hint}
            >
              <div className={cn("flex items-center gap-1 text-[10px] uppercase tracking-wide", meta.tint)}>
                <Icon className="h-3 w-3" />
                {meta.label}
              </div>
              <div className="mt-1 font-mono text-base font-semibold leading-none">
                {count}
              </div>
            </div>
          );
        })}
      </div>

      <div className="space-y-1 px-1">
        <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
          Recent writes
        </div>
        {entries.length > 0 ? (
          <>
            {entries.map((e, i) => (
              <MemoryRow key={`${e.ts_ms}-${e.scope}-${i}`} entry={e} now={now} />
            ))}
            {hasMore && (
              <button
                type="button"
                onClick={() => setLimit((n) => n + PAGE_STEP)}
                className="w-full rounded-md border border-dashed bg-card/30 px-2 py-1.5 text-[11px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                Load more
              </button>
            )}
          </>
        ) : showEmptyState ? (
          <div className="rounded-md border border-dashed bg-card/30 px-2 py-3 text-[11px] text-muted-foreground">
            {contextId
              ? "When the agent saves something to remember later, it shows up here. Try asking it to remember a fact about you."
              : "Connecting to session…"}
          </div>
        ) : (
          <div
            aria-hidden="true"
            className="h-9 rounded-md border border-dashed bg-card/20"
          />
        )}
      </div>
    </div>
  );
}

const MemoryRow = memo(function MemoryRow({
  entry,
  now,
}: {
  entry: HorizonMemoryEntry;
  now: number;
}) {
  const meta = SCOPE_META[entry.scope] ?? SCOPE_META.agent;
  const Icon = meta.Icon;
  return (
    <div className="rounded-md border bg-card/50 px-2 py-1.5 text-[11px] motion-safe:animate-fade-in-up">
      <div className="flex items-center justify-between gap-2">
        <span className={cn("inline-flex items-center gap-1 text-[10px] uppercase", meta.tint)}>
          <Icon className="h-2.5 w-2.5" />
          {meta.label}
        </span>
        <span className="font-mono text-[10px] text-muted-foreground">
          {timeAgo(entry.ts_ms, now)}
        </span>
      </div>
      <div className="mt-1 leading-snug text-foreground/90">{entry.content}</div>
    </div>
  );
});
