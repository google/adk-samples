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

import type { ReactNode } from "react";
import { ChevronRight, MessageSquarePlus, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface ProjectSectionProps {
  name: string;
  // Optional: the lazy Scheduled group has no count until it's opened.
  count?: number;
  collapsed: boolean;
  onToggle: () => void;
  icon?: ReactNode;
  // Omitted for the General / Scheduled buckets (no folder to create in / delete).
  onNewChat?: () => void;
  onRequestDelete?: () => void;
  children: ReactNode;
}

export function ProjectSection({
  name,
  count,
  collapsed,
  onToggle,
  icon,
  onNewChat,
  onRequestDelete,
  children,
}: ProjectSectionProps) {
  return (
    <section className="flex flex-col">
      <div className="group/proj flex items-center gap-0.5 px-1">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={!collapsed}
          className="flex min-w-0 flex-1 items-center gap-1 rounded-md px-1 py-1 text-left text-xs font-semibold text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <ChevronRight
            className={cn(
              "h-3.5 w-3.5 shrink-0 text-muted-foreground/60 transition-transform",
              !collapsed && "rotate-90",
            )}
          />
          {icon && (
            <span className="shrink-0 text-muted-foreground/70">{icon}</span>
          )}
          <span className="truncate">{name}</span>
          {count !== undefined && (
            <span className="shrink-0 text-[10px] font-normal text-muted-foreground/50">
              {count}
            </span>
          )}
        </button>
        {onNewChat && (
          <button
            type="button"
            onClick={onNewChat}
            aria-label={`New chat in ${name}`}
            title="New chat"
            className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-muted-foreground opacity-0 transition-opacity hover:bg-accent hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/proj:opacity-100"
          >
            <MessageSquarePlus className="h-3.5 w-3.5" />
          </button>
        )}
        {onRequestDelete && (
          <button
            type="button"
            onClick={onRequestDelete}
            aria-label={`Delete project ${name}`}
            title="Delete project"
            className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/15 hover:text-destructive focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/proj:opacity-100"
          >
            <Trash2 className="h-3 w-3" />
          </button>
        )}
      </div>
      {!collapsed && <div className="pb-1">{children}</div>}
    </section>
  );
}
