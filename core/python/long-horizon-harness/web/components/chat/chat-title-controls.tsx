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


import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { Pencil, Trash2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  deleteLhaSession,
  renameLhaSession,
} from "@/lib/horizon-sessions";
import { cn } from "@/lib/utils";

interface UseChatTitleControlsArgs {
  sessionId: string;
  title: string;
  onAfterRename?: () => void;
  onAfterDelete?: () => void;
}

export interface ChatTitleControls {
  editing: boolean;
  startEditing: () => void;
  draft: string;
  setDraft: (s: string) => void;
  saving: boolean;
  commit: () => Promise<void>;
  cancel: () => void;
  inputRef: RefObject<HTMLInputElement>;
  deleteOpen: boolean;
  setDeleteOpen: (b: boolean) => void;
  deleting: boolean;
  deleteError: string | null;
  confirmDelete: () => Promise<void>;
}

export function useChatTitleControls({
  sessionId,
  title,
  onAfterRename,
  onAfterDelete,
}: UseChatTitleControlsArgs): ChatTitleControls {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title);
  const [saving, setSaving] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Reset on session switch only — a refetch of the same session must not clobber an in-progress edit.
  useEffect(() => {
    setEditing(false);
    setDraft(title);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  useEffect(() => {
    if (!editing) setDraft(title);
  }, [title, editing]);

  useEffect(() => {
    if (!editing) return;
    requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });
  }, [editing]);

  useEffect(() => {
    if (!deleteError) return;
    const t = setTimeout(() => setDeleteError(null), 5000);
    return () => clearTimeout(t);
  }, [deleteError]);

  const commit = async () => {
    const trimmed = draft.trim();
    if (!trimmed || trimmed === title) {
      setEditing(false);
      setDraft(title);
      return;
    }
    setSaving(true);
    try {
      await renameLhaSession(sessionId, trimmed);
      onAfterRename?.();
    } catch {
      setDraft(title);
    } finally {
      setSaving(false);
      setEditing(false);
    }
  };

  const cancel = () => {
    setDraft(title);
    setEditing(false);
  };

  const confirmDelete = async () => {
    setDeleting(true);
    try {
      await deleteLhaSession(sessionId);
      setDeleteOpen(false);
      setDeleteError(null);
      onAfterDelete?.();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : String(err));
      setDeleteOpen(false);
    } finally {
      setDeleting(false);
    }
  };

  return {
    editing,
    startEditing: () => setEditing(true),
    draft,
    setDraft,
    saving,
    commit,
    cancel,
    inputRef,
    deleteOpen,
    setDeleteOpen,
    deleting,
    deleteError,
    confirmDelete,
  };
}

interface ChatTitleInputProps {
  controls: ChatTitleControls;
  className?: string;
}

export function ChatTitleInput({ controls, className }: ChatTitleInputProps) {
  return (
    <Input
      ref={controls.inputRef}
      value={controls.draft}
      disabled={controls.saving}
      maxLength={80} // mirrors the backend's TITLE_MAX_LEN, so a rename can't 422
      onChange={(e) => controls.setDraft(e.target.value)}
      onBlur={() => void controls.commit()}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          void controls.commit();
        } else if (e.key === "Escape") {
          e.preventDefault();
          controls.cancel();
        }
      }}
      className={className}
    />
  );
}

interface ChatActionButtonsProps {
  title: string;
  controls: ChatTitleControls;
  stopPropagation?: boolean;
  dialogContent?: ReactNode;
}

export function ChatActionButtons({
  title,
  controls,
  stopPropagation = false,
  dialogContent,
}: ChatActionButtonsProps) {
  const stop = (e: React.MouseEvent) => {
    if (stopPropagation) e.stopPropagation();
  };
  return (
    <>
      <button
        type="button"
        onClick={(e) => {
          stop(e);
          controls.startEditing();
        }}
        className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
        aria-label="Rename chat"
      >
        <Pencil className="h-3.5 w-3.5" />
      </button>
      <AlertDialog open={controls.deleteOpen} onOpenChange={controls.setDeleteOpen}>
        <AlertDialogTrigger asChild>
          <button
            type="button"
            onClick={(e) => {
              stop(e);
              controls.setDeleteOpen(true);
            }}
            className="rounded p-1 text-muted-foreground hover:bg-destructive/20 hover:text-destructive"
            aria-label="Delete chat"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </AlertDialogTrigger>
        {controls.deleteOpen && (
          <AlertDialogContent onClick={stopPropagation ? (e) => e.stopPropagation() : undefined}>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete chat?</AlertDialogTitle>
              <AlertDialogDescription>
                Delete &ldquo;{title}&rdquo;? This can&rsquo;t be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            {dialogContent}
            <AlertDialogFooter>
              <AlertDialogCancel disabled={controls.deleting}>Cancel</AlertDialogCancel>
              <AlertDialogAction
                disabled={controls.deleting}
                onClick={(e) => {
                  e.preventDefault();
                  void controls.confirmDelete();
                }}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        )}
      </AlertDialog>
    </>
  );
}

interface ChatDeleteErrorProps {
  error: string | null;
  className?: string;
}

export function ChatDeleteError({ error, className }: ChatDeleteErrorProps) {
  if (!error) return null;
  return (
    <div
      role="alert"
      className={cn(
        "rounded border border-destructive/40 bg-destructive/10 px-2 py-1 text-[11px] text-destructive",
        className,
      )}
    >
      Failed to delete: {error}
    </div>
  );
}
