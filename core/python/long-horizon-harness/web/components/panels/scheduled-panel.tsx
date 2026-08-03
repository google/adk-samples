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

import { memo, type ReactNode } from "react";
import { RemindersPanel } from "@/components/panels/reminders-panel";
import { RoutinesPanel } from "@/components/panels/routines-panel";

function Sub({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <p className="px-2 pt-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground/70">
        {label}
      </p>
      {children}
    </div>
  );
}

function ScheduledPanelImpl() {
  return (
    <div className="flex flex-col gap-1">
      <Sub label="Reminders">
        <RemindersPanel />
      </Sub>
      <Sub label="Routines">
        <RoutinesPanel />
      </Sub>
    </div>
  );
}

export const ScheduledPanel = memo(ScheduledPanelImpl);
