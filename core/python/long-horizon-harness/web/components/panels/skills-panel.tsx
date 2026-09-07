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


import { Sparkles } from "lucide-react";
import {
  useLhaState,
  type HorizonBoundSkill,
  type HorizonStateResponse,
} from "@/lib/horizon-state";

interface SkillsPanelProps {
  contextId: string | null;
}

interface SkillCounter {
  views?: number;
  manages?: number;
  last_used?: string | null;
}

const selectSkills = (d: HorizonStateResponse) => ({
  skills: d.state.bound_skills ?? [],
  telemetry: (d.state.skill_telemetry ?? {}) as Record<string, SkillCounter>,
});

function sortSkills(
  skills: HorizonBoundSkill[],
  telemetry: Record<string, SkillCounter>,
): HorizonBoundSkill[] {
  const views = (name: string) => telemetry[name]?.views ?? 0;
  return [...skills].sort((a, b) => {
    if (a.source !== b.source) return a.source === "user" ? -1 : 1;
    const dv = views(b.name) - views(a.name);
    if (dv !== 0) return dv;
    return a.name.localeCompare(b.name);
  });
}

export function SkillsPanel({ contextId }: SkillsPanelProps) {
  const { data } = useLhaState(contextId, { select: selectSkills });
  const skills = data?.skills ?? [];
  const telemetry = data?.telemetry ?? {};
  const ordered = sortSkills(skills, telemetry);

  return (
    <div className="space-y-1 px-1">
      {ordered.length === 0 ? (
        <div className="rounded-md border border-dashed bg-card/30 px-2 py-3 text-[11px] text-muted-foreground">
          {contextId
            ? "Skills are reusable playbooks the agent can load on demand. None installed yet."
            : "Connecting to session…"}
        </div>
      ) : (
        ordered.map((skill) => {
          const views = telemetry[skill.name]?.views ?? 0;
          return (
            <div
              key={skill.name}
              className="flex items-center justify-between gap-2 rounded-md border bg-card/50 px-2 py-1.5 text-[11px]"
              title={skill.description || undefined}
            >
              <div className="flex min-w-0 items-center gap-1.5">
                <Sparkles className="h-3 w-3 shrink-0 text-[hsl(var(--lh-warp))]" />
                <span className="truncate font-mono">{skill.name}</span>
              </div>
              <div className="flex shrink-0 items-center gap-1.5 font-mono text-[10px] text-muted-foreground">
                <span
                  className={
                    skill.source === "user"
                      ? "text-[hsl(var(--lh-warp))]"
                      : "opacity-60"
                  }
                >
                  {skill.source === "user" ? "user" : "built-in"}
                </span>
                {views > 0 ? <span>{views}×</span> : null}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
