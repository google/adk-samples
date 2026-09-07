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
import { Trash2 } from "lucide-react";
import {
  useProcesses,
  killProcess,
  type HorizonProcess,
} from "@/lib/horizon-processes";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { cn } from "@/lib/utils";

function BackgroundPanelImpl() {
  const { data, isLoading, refresh } = useProcesses();
  const [pending, setPending] = useState<HorizonProcess | null>(null);
  const [killing, setKilling] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // The endpoint returns finished handles too (their output stays readable via
  // the `process` tool), but this panel is live work you can kill — and the
  // section count already filters the same way.
  const procs = (data?.processes ?? []).filter((p) => p.running);
  const showEmpty = !isLoading && procs.length === 0;

  async function confirmKill() {
    if (!pending) return;
    setKilling(true);
    setErr(null);
    try {
      await killProcess(pending.session_id);
      await refresh();
      setPending(null);
    } catch {
      setErr("Failed to kill process.");
    } finally {
      setKilling(false);
    }
  }

  return (
    <div className="flex flex-col gap-2 p-2 text-sm">
      {showEmpty && (
        <p className="text-xs text-muted-foreground">No background processes.</p>
      )}

      <ul className="flex flex-col gap-1">
        {procs.map((p) => (
          <li
            key={p.session_id}
            className={cn(
              "flex items-start justify-between gap-2 rounded px-2 py-1 hover:bg-muted",
              !p.running && "opacity-60",
            )}
          >
            <span className="flex min-w-0 flex-col gap-0.5">
              <span className="truncate font-mono text-xs">{p.command}</span>
              <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                <span
                  className={cn(
                    "inline-block h-1.5 w-1.5 rounded-full",
                    p.running ? "bg-emerald-500" : "bg-muted-foreground/40",
                  )}
                />
                {p.running
                  ? `idle ${Math.round(p.idle_seconds)}s`
                  : `exit ${p.exit_code ?? "?"}`}
              </span>
            </span>
            <button
              type="button"
              aria-label={`Kill process: ${p.command}`}
              className="mt-0.5 text-muted-foreground hover:text-red-500"
              onClick={() => setPending(p)}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </li>
        ))}
      </ul>

      {err && <p className="text-xs text-red-500">{err}</p>}

      <AlertDialog
        open={pending !== null}
        onOpenChange={(o) => {
          if (!o) {
            setPending(null);
            setErr(null);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Kill this process?</AlertDialogTitle>
            <AlertDialogDescription>
              {pending?.command
                ? `"${pending.command}" will be terminated.`
                : "This process will be terminated."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={killing}>Keep</AlertDialogCancel>
            <AlertDialogAction
              disabled={killing}
              onClick={(e) => {
                e.preventDefault();
                void confirmKill();
              }}
              className={cn(killing && "opacity-50")}
            >
              Kill
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export const BackgroundPanel = memo(BackgroundPanelImpl);
