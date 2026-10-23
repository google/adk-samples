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


import { useMemo, useState } from "react";
import { CalendarClock } from "lucide-react";
import { useScheduledSessions } from "@/lib/horizon-sessions";
import { groupByJobType } from "@/lib/scheduled-groups";
import { ProjectSection } from "./project-section";
import { SidebarSearchInput } from "./sidebar-search-input";
import { SidebarSessionGroups } from "./sidebar-session-groups";

// A pseudo-project group: same expander styling as real projects, but not
// folder-backed. Default collapsed + lazy (its own persisted open state, kept
// out of the shared project-collapse set so its default stays collapsed).
const SCHED_OPEN_KEY = "lha.sidebar.scheduled.open";

function loadOpen(): boolean {
  try {
    return localStorage.getItem(SCHED_OPEN_KEY) === "1";
  } catch {
    return false;
  }
}

interface ScheduledSectionProps {
  activeContextId: string | null;
  onSelect: (id: string) => void;
  onAfterRename: () => void;
  onAfterDelete: (id: string) => void;
  onPrefetch?: (id: string) => void;
}

export function ScheduledSection({
  activeContextId,
  onSelect,
  onAfterRename,
  onAfterDelete,
  onPrefetch,
}: ScheduledSectionProps) {
  // Lazy: only fetch once the user opens the group. Default collapsed because
  // scheduled sessions can be numerous and aren't the primary surface.
  const [open, setOpen] = useState(loadOpen);
  const [query, setQuery] = useState("");
  const { data, isLoading } = useScheduledSessions(open, query);

  const groups = useMemo(() => groupByJobType(data ?? []), [data]);

  const toggle = () => {
    setOpen((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(SCHED_OPEN_KEY, next ? "1" : "0");
      } catch {
        // best-effort persistence
      }
      return next;
    });
  };

  return (
    <ProjectSection
      name="Scheduled"
      icon={<CalendarClock className="h-3.5 w-3.5" />}
      count={data?.length}
      collapsed={!open}
      onToggle={toggle}
    >
      <div className="px-1 pt-1">
        <SidebarSearchInput
          value={query}
          onChange={setQuery}
          placeholder="Search scheduled…"
          ariaLabel="Search scheduled sessions"
        />
        <div className="max-h-[40vh] overflow-y-auto">
          {isLoading && !data ? (
            <div className="px-2 py-4 text-center text-xs text-muted-foreground">
              Loading…
            </div>
          ) : groups.length === 0 ? (
            <div className="px-2 py-4 text-center text-xs text-muted-foreground">
              {query ? "No matches." : "No scheduled sessions yet."}
            </div>
          ) : (
            <SidebarSessionGroups
              groups={groups}
              activeContextId={activeContextId}
              onSelect={onSelect}
              onAfterRename={onAfterRename}
              onAfterDelete={onAfterDelete}
              onPrefetch={onPrefetch}
            />
          )}
        </div>
      </div>
    </ProjectSection>
  );
}
