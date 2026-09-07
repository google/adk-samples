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
import { CalendarClock, Repeat, Trash2 } from "lucide-react";
import {
  useRoutines,
  deleteRoutine,
  type HorizonRoutine,
} from "@/lib/horizon-routines";
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

function formatNextFire(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function RoutinesPanelImpl() {
  const { data, isLoading, refresh } = useRoutines();
  const [pending, setPending] = useState<HorizonRoutine | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const routines = data?.routines ?? [];
  const showEmpty = !isLoading && routines.length === 0;

  async function confirmCancel() {
    if (!pending) return;
    setDeleting(true);
    setErr(null);
    try {
      await deleteRoutine(pending.id);
      await refresh();
      setPending(null);
    } catch {
      setErr("Failed to cancel routine.");
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="flex flex-col gap-2 p-2 text-sm">
      {showEmpty && (
        <p className="text-xs text-muted-foreground">
          No routines scheduled. Ask the agent to run a recurring task for you.
        </p>
      )}

      <ul className="flex flex-col gap-1">
        {routines.map((r) => (
          <li
            key={r.id}
            className="flex items-start justify-between gap-2 rounded px-2 py-1 hover:bg-muted"
          >
            <span className="flex flex-col gap-0.5">
              <span className="text-xs">{r.task}</span>
              <span className="flex items-center gap-2 text-[10px] text-muted-foreground">
                <span className="flex items-center gap-0.5">
                  <Repeat className="h-3 w-3" />
                  {r.schedule}
                </span>
                <span className="flex items-center gap-0.5">
                  <CalendarClock className="h-3 w-3" />
                  {formatNextFire(r.next_fire_at)}
                </span>
              </span>
            </span>
            <button
              type="button"
              aria-label={`Cancel routine: ${r.task}`}
              className="mt-0.5 text-muted-foreground hover:text-red-500"
              onClick={() => setPending(r)}
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
            <AlertDialogTitle>Cancel this routine?</AlertDialogTitle>
            <AlertDialogDescription>
              {pending?.task
                ? `"${pending.task}" will no longer run. This can't be undone.`
                : "This routine will no longer run. This can't be undone."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Keep</AlertDialogCancel>
            <AlertDialogAction
              disabled={deleting}
              onClick={(e) => {
                e.preventDefault();
                void confirmCancel();
              }}
              className={cn(deleting && "opacity-50")}
            >
              Cancel routine
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export const RoutinesPanel = memo(RoutinesPanelImpl);
