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

import { useState } from "react";
import { Folder, FolderPlus, MessageSquarePlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

interface NewConversationMenuProps {
  projects: string[];
  onGeneral: () => void;
  onPickProject: (name: string) => void;
  onCreateProject: (name: string) => Promise<void>;
}

const ROW =
  "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent focus-visible:bg-accent focus-visible:outline-none";

export function NewConversationMenu({
  projects,
  onGeneral,
  onPickProject,
  onCreateProject,
}: NewConversationMenuProps) {
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const close = () => {
    setOpen(false);
    setCreating(false);
    setName("");
    setError(null);
  };

  const submitCreate = async () => {
    const trimmed = name.trim();
    if (!trimmed || trimmed.includes("/") || trimmed === "." || trimmed === "..") {
      setError("Enter a valid folder name (no “/”).");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onCreateProject(trimmed);
      close();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Popover open={open} onOpenChange={(o) => (o ? setOpen(true) : close())}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          className="nc-trigger h-9 w-full justify-start gap-2 text-sm font-medium"
          title="New Conversation"
        >
          <MessageSquarePlus className="h-4 w-4 shrink-0" />
          <span className="nc-label truncate">New Conversation</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" sideOffset={6} className="w-64 p-1">
        <button
          type="button"
          className={ROW}
          onClick={() => {
            onGeneral();
            close();
          }}
        >
          <MessageSquarePlus className="h-4 w-4 text-muted-foreground" />
          General chat
        </button>

        {projects.length > 0 && (
          <>
            <div className="px-2 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/60">
              Projects
            </div>
            <div className="max-h-48 overflow-y-auto">
              {projects.map((p) => (
                <button
                  key={p}
                  type="button"
                  className={ROW}
                  onClick={() => {
                    onPickProject(p);
                    close();
                  }}
                >
                  <Folder className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{p}</span>
                </button>
              ))}
            </div>
          </>
        )}

        <div className="my-1 border-t" />

        {!creating ? (
          <button
            type="button"
            className={`${ROW} text-muted-foreground hover:text-foreground`}
            onClick={() => setCreating(true)}
          >
            <FolderPlus className="h-4 w-4" />
            New project…
          </button>
        ) : (
          <div className="flex flex-col gap-1 p-1">
            {/* eslint-disable-next-line jsx-a11y/no-autofocus */}
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void submitCreate();
                } else if (e.key === "Escape") {
                  setCreating(false);
                  setName("");
                  setError(null);
                }
              }}
              disabled={busy}
              placeholder="Project name"
              aria-label="Project name"
              className="h-8 w-full rounded-md border bg-background px-2 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
            {error && (
              <div className="px-1 text-[10px] text-destructive">{error}</div>
            )}
            <Button
              type="button"
              size="sm"
              onClick={() => void submitCreate()}
              disabled={busy}
              className="h-7 text-[11px]"
            >
              {busy ? "Creating…" : "Create & start chat"}
            </Button>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
