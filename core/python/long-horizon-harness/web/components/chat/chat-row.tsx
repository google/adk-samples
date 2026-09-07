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


import { memo, useCallback, useEffect, useRef } from "react";
import { useNow } from "@/lib/now-context";
import { cn } from "@/lib/utils";
import type { HorizonSessionSummary } from "@/lib/horizon-sessions";
import {
  ChatActionButtons,
  ChatDeleteError,
  ChatTitleInput,
  useChatTitleControls,
} from "./chat-title-controls";

export const DAY_MS = 86_400_000;

export function formatRelative(ts: number, now: number): string {
  const diff = now - ts;
  if (diff < 60_000) return "now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m`;
  if (diff < DAY_MS) return `${Math.floor(diff / 3_600_000)}h`;
  if (diff < 2 * DAY_MS) return "1d";
  if (diff < 7 * DAY_MS) return `${Math.floor(diff / DAY_MS)}d`;
  return new Date(ts).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export interface ChatRowProps {
  session: HorizonSessionSummary;
  active: boolean;
  onSelect: (id: string) => void;
  onAfterRename: () => void;
  onAfterDelete: (id: string) => void;
  // Warm this chat's history after a brief hover so opening it feels instant.
  onPrefetch?: (id: string) => void;
}

// Hover this long before prefetching so cursoring past rows doesn't fire a
// request for every chat the pointer crosses.
const PREFETCH_HOVER_MS = 150;

export const ChatRow = memo(function ChatRow({
  session,
  active,
  onSelect,
  onAfterRename,
  onAfterDelete,
  onPrefetch,
}: ChatRowProps) {
  const now = useNow();
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cancelHover = useCallback(() => {
    if (hoverTimer.current) {
      clearTimeout(hoverTimer.current);
      hoverTimer.current = null;
    }
  }, []);

  const handleMouseEnter = useCallback(() => {
    if (active || !onPrefetch) return;
    cancelHover();
    hoverTimer.current = setTimeout(
      () => onPrefetch(session.id),
      PREFETCH_HOVER_MS,
    );
  }, [active, onPrefetch, cancelHover, session.id]);

  useEffect(() => cancelHover, [cancelHover]);

  const handleAfterDelete = useCallback(
    () => onAfterDelete(session.id),
    [onAfterDelete, session.id],
  );
  const controls = useChatTitleControls({
    sessionId: session.id,
    title: session.title,
    onAfterRename,
    onAfterDelete: handleAfterDelete,
  });

  const navigate = () => {
    if (controls.editing) return;
    onSelect(session.id);
  };

  return (
    <li>
      <div
        role="button"
        tabIndex={0}
        onClick={navigate}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={cancelHover}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            navigate();
          }
        }}
        onDoubleClick={(e) => {
          e.stopPropagation();
          controls.startEditing();
        }}
        className={cn(
          "group relative flex items-center gap-1.5 rounded-md px-2 py-2 text-sm transition-colors",
          active
            ? "bg-accent font-medium text-foreground before:absolute before:left-0 before:top-1 before:bottom-1 before:w-1 before:rounded-full before:bg-primary"
            : "lh-sidebar-hover text-muted-foreground",
          controls.editing && "bg-accent/40",
        )}
      >
        {controls.editing ? (
          <ChatTitleInput
            controls={controls}
            className="h-6 px-1.5 py-0 text-sm"
          />
        ) : (
          <>
            <span className="min-w-0 flex-1 truncate pr-12" title={session.title}>
              {session.title}
            </span>
            <span className="absolute right-2 top-1/2 -translate-y-1/2 shrink-0 font-mono text-[10px] text-muted-foreground/60 transition-opacity group-hover:opacity-0 max-md:right-12 max-md:group-hover:opacity-100">
              {formatRelative(session.lastUpdated, now)}
            </span>
            <div className="pointer-events-none absolute right-1.5 top-1/2 flex -translate-y-1/2 items-center gap-0.5 opacity-0 transition-opacity group-hover:pointer-events-auto group-hover:opacity-100 focus-within:pointer-events-auto focus-within:opacity-100 max-md:pointer-events-auto max-md:opacity-100">
              <ChatActionButtons
                title={session.title}
                controls={controls}
                stopPropagation
              />
            </div>
          </>
        )}
      </div>
      <ChatDeleteError error={controls.deleteError} className="mx-2 mt-1" />
    </li>
  );
});
