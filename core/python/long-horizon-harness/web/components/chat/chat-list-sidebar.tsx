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


import { memo, useCallback, useMemo, useRef, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { RefreshCw, Trash2 } from "lucide-react";
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
import {
  deleteAllLhaSessions,
  useLhaSessions,
  type HorizonSessionSummary,
} from "@/lib/horizon-sessions";
import {
  createProjectFolder,
  deleteProject,
  groupSessionsByProject,
  useWorkspaceFolders,
} from "@/lib/projects";
import { setPendingWindow } from "@/lib/pending-window";
import { loadCollapsed, persistCollapsed } from "@/lib/project-collapse";
import {
  addCreatedProject,
  loadCreatedProjects,
  removeCreatedProject,
} from "@/lib/project-created";
import { ChatRow, DAY_MS } from "./chat-row";
import { ProjectSection } from "./project-section";
import { NewConversationMenu } from "./new-conversation-menu";
import { DeleteProjectDialog } from "./delete-project-dialog";
import { ScheduledSection } from "./scheduled-section";
import { SidebarSection } from "./sidebar-section";
import { SidebarSearchInput } from "./sidebar-search-input";
import { SidebarSessionGroups } from "./sidebar-session-groups";
import { SidebarDropzone, type UploadResult } from "./sidebar-dropzone";
import {
  SidebarUploadList,
  type RecentUpload,
} from "./sidebar-upload-list";
import {
  SidebarWorkspaceTree,
  type OptimisticRow,
} from "./sidebar-workspace-tree";

interface ChatListSidebarProps {
  activeContextId: string | null;
  sessions: HorizonSessionSummary[] | undefined;
  sessionsLoading: boolean;
  refreshSessions: () => Promise<unknown>;
  onNavigate?: () => void;
  // Called instead of router.push("/") so the parent can reset in-page
  // state when we're already on the home route (no nav → no remount).
  onNewChat?: () => void;
  // Warm a chat's history on hover so opening it is instant.
  onPrefetch?: (id: string) => void;
  // When true, reserve room on the first section header for a sheet close
  // button (mobile drawer rendering).
  reserveHeaderRight?: boolean;
  // Upload + workspace wiring (owned by chat-shell).
  recentUploads: RecentUpload[];
  optimisticRows: OptimisticRow[];
  workspaceVersion: number;
  onUploadComplete: (result: UploadResult) => void;
  onWorkspaceChanged: () => void;
  onReconciled: () => void;
  onClearRecent: () => void;
  onDeleteWorkspaceFile: (
    path: string,
    opts?: { recursive?: boolean },
  ) => Promise<void>;
}

interface SessionGroup {
  label: string;
  sessions: HorizonSessionSummary[];
}

function groupSessions(sessions: HorizonSessionSummary[]): SessionGroup[] {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const yesterdayStart = todayStart - DAY_MS;
  const week = todayStart - 7 * DAY_MS;
  const month = todayStart - 30 * DAY_MS;

  const buckets: Record<string, HorizonSessionSummary[]> = {
    Today: [],
    Yesterday: [],
    "Last 7 days": [],
    "Last 30 days": [],
    Older: [],
  };

  const sorted = [...sessions].sort((a, b) => b.lastUpdated - a.lastUpdated);
  for (const s of sorted) {
    const ts = s.lastUpdated;
    if (ts >= todayStart) buckets.Today.push(s);
    else if (ts >= yesterdayStart) buckets.Yesterday.push(s);
    else if (ts >= week) buckets["Last 7 days"].push(s);
    else if (ts >= month) buckets["Last 30 days"].push(s);
    else buckets.Older.push(s);
  }

  return Object.entries(buckets)
    .filter(([, list]) => list.length > 0)
    .map(([label, list]) => ({ label, sessions: list }));
}

function ChatListSidebarInner({
  activeContextId,
  sessions,
  sessionsLoading,
  refreshSessions,
  onNavigate,
  onNewChat,
  onPrefetch,
  reserveHeaderRight,
  recentUploads,
  optimisticRows,
  workspaceVersion,
  onUploadComplete,
  onWorkspaceChanged,
  onReconciled,
  onClearRecent,
  onDeleteWorkspaceFile,
}: ChatListSidebarProps) {
  const navigate = useNavigate();

  // Chat search hits the backend (?source=user&q=) so it finds chats beyond
  // the loaded page; an empty query reuses the parent's list (same query key).
  const [chatQuery, setChatQuery] = useState("");
  const searching = chatQuery.trim().length > 0;
  const { data: searchData, isLoading: searchLoading } = useLhaSessions(chatQuery);
  const chatList = searching ? searchData : sessions;
  const chatLoading = searching ? searchLoading : sessionsLoading;

  const { data: folders, refresh: refreshFolders } = useWorkspaceFolders();
  const [collapsed, setCollapsed] = useState<Set<string>>(() => loadCollapsed());
  const [deleteTarget, setDeleteTarget] = useState<{
    name: string;
    chatIds: string[];
  } | null>(null);
  const [createdProjects, setCreatedProjects] =
    useState<Set<string>>(loadCreatedProjects);
  const workspaceRefreshRef = useRef<(() => void) | null>(null);
  const registerWorkspaceRefresh = useCallback((fn: () => void) => {
    workspaceRefreshRef.current = fn;
  }, []);
  const toggleProject = useCallback((name: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      persistCollapsed(next);
      return next;
    });
  }, []);

  // A chat's project = its workspace_window anchor; empty => General. Only
  // introduce the project expanders once there IS a project — otherwise the
  // list stays the familiar flat date-grouped view.
  const grouped = useMemo(
    () =>
      groupSessionsByProject(
        chatList ?? [],
        searching ? [] : (folders ?? []),
        createdProjects,
      ),
    [chatList, searching, folders, createdProjects],
  );
  const hasProjects = grouped.projects.length > 0;
  const generalGroups = useMemo(
    () => groupSessions(grouped.general),
    [grouped.general],
  );

  const handleNewProjectChat = useCallback(
    (name: string) => {
      const id = crypto.randomUUID();
      setPendingWindow(id, [name]);
      navigate({ to: "/c", search: { id } });
      onNavigate?.();
    },
    [navigate, onNavigate],
  );

  const handleCreateProjectAndChat = useCallback(
    async (name: string) => {
      await createProjectFolder(name);
      setCreatedProjects(addCreatedProject(name));
      await Promise.all([refreshFolders(), refreshSessions()]);
      handleNewProjectChat(name);
    },
    [refreshFolders, refreshSessions, handleNewProjectChat],
  );

  const handleConfirmDelete = useCallback(async () => {
    if (!deleteTarget) return;
    const { name, chatIds } = deleteTarget;
    await deleteProject(name, chatIds);
    setCreatedProjects(removeCreatedProject(name));
    await Promise.all([refreshFolders(), refreshSessions()]);
    if (activeContextId && chatIds.includes(activeContextId)) {
      navigate({ to: "/" });
    }
  }, [deleteTarget, refreshFolders, refreshSessions, activeContextId, navigate]);

  const goNew = () => {
    if (onNewChat) {
      onNewChat();
    } else {
      navigate({ to: "/" });
    }
    onNavigate?.();
  };

  const handleSelect = useCallback(
    (id: string) => {
      navigate({ to: "/c", search: { id } });
      onNavigate?.();
    },
    [navigate, onNavigate],
  );

  const handleAfterRename = useCallback(() => {
    void refreshSessions();
  }, [refreshSessions]);

  const handleAfterDelete = useCallback(
    (id: string) => {
      void refreshSessions();
      if (id === activeContextId) navigate({ to: "/" });
    },
    [refreshSessions, activeContextId, navigate],
  );

  return (
    <nav
      aria-label="Sessions"
      className="sidebar-cq flex h-full min-h-0 flex-col"
    >
      <div className="p-2">
        <NewConversationMenu
          projects={grouped.projects.map((p) => p.name)}
          onGeneral={goNew}
          onPickProject={handleNewProjectChat}
          onCreateProject={handleCreateProjectAndChat}
        />
      </div>
      <SidebarSection
        title="Chats"
        storageKey="lha.sidebar.chats"
        defaultOpen
        headerClassName={reserveHeaderRight ? "pr-12" : undefined}
        rightSlot={
          sessions && sessions.length > 0 ? (
            <DeleteAllChatsButton
              count={sessions.length}
              onAfterDelete={() => {
                void refreshSessions();
                // If the user is sitting on a chat that just got nuked,
                // bounce them home so they don't see a phantom session.
                if (activeContextId) navigate({ to: "/" });
              }}
            />
          ) : undefined
        }
      >
        <div className="px-2 py-2">
          <SidebarSearchInput
            value={chatQuery}
            onChange={setChatQuery}
            placeholder="Search chats…"
            ariaLabel="Search chats"
          />
          <div className="max-h-[40vh] overflow-y-auto">
            {chatLoading && !chatList ? (
              <SidebarSkeleton />
            ) : !chatList || chatList.length === 0 ? (
              <div className="px-2 py-6 text-center text-xs text-muted-foreground">
                {searching ? (
                  "No matching chats."
                ) : (
                  <>
                    No chats yet.
                    <br />
                    Send a message to start one.
                  </>
                )}
              </div>
            ) : !hasProjects ? (
              <SidebarSessionGroups
                groups={generalGroups}
                activeContextId={activeContextId}
                onSelect={handleSelect}
                onAfterRename={handleAfterRename}
                onAfterDelete={handleAfterDelete}
                onPrefetch={onPrefetch}
              />
            ) : (
              <div className="flex flex-col gap-1">
                {grouped.general.length > 0 && (
                  <ProjectSection
                    name="General"
                    count={grouped.general.length}
                    collapsed={collapsed.has("General")}
                    onToggle={() => toggleProject("General")}
                  >
                    <SidebarSessionGroups
                      groups={generalGroups}
                      activeContextId={activeContextId}
                      onSelect={handleSelect}
                      onAfterRename={handleAfterRename}
                      onAfterDelete={handleAfterDelete}
                      onPrefetch={onPrefetch}
                    />
                  </ProjectSection>
                )}
                {grouped.projects.map((p) => (
                  <ProjectSection
                    key={p.name}
                    name={p.name}
                    count={p.chats.length}
                    collapsed={collapsed.has(p.name)}
                    onToggle={() => toggleProject(p.name)}
                    onNewChat={() => handleNewProjectChat(p.name)}
                    onRequestDelete={() =>
                      setDeleteTarget({
                        name: p.name,
                        chatIds: p.chats.map((c) => c.id),
                      })
                    }
                  >
                    {p.chats.length === 0 ? (
                      <div className="px-4 py-1.5 text-[11px] text-muted-foreground/50">
                        No chats yet.
                      </div>
                    ) : (
                      <ul className="flex flex-col gap-0.5">
                        {p.chats.map((s) => (
                          <ChatRow
                            key={s.id}
                            session={s}
                            active={s.id === activeContextId}
                            onSelect={handleSelect}
                            onAfterRename={handleAfterRename}
                            onAfterDelete={handleAfterDelete}
                            onPrefetch={onPrefetch}
                          />
                        ))}
                      </ul>
                    )}
                  </ProjectSection>
                ))}
              </div>
            )}
          </div>
          <ScheduledSection
            activeContextId={activeContextId}
            onSelect={handleSelect}
            onAfterRename={handleAfterRename}
            onAfterDelete={handleAfterDelete}
            onPrefetch={onPrefetch}
          />
        </div>
      </SidebarSection>

      <SidebarSection
        title="Upload"
        storageKey="lha.sidebar.dropzone"
        defaultOpen={false}
      >
        <SidebarDropzone
          onUploadComplete={onUploadComplete}
          onWorkspaceChanged={onWorkspaceChanged}
        />
      </SidebarSection>

      {recentUploads.length > 0 && (
        <SidebarSection
          title="Recent uploads (this session)"
          storageKey="lha.sidebar.recent"
          defaultOpen
          rightSlot={
            <button
              type="button"
              onClick={onClearRecent}
              className="text-[10px] text-muted-foreground/70 hover:text-foreground"
            >
              Clear
            </button>
          }
        >
          <SidebarUploadList items={recentUploads} />
        </SidebarSection>
      )}

      <SidebarSection
        title="Workspace"
        storageKey="lha.sidebar.workspace"
        defaultOpen
        collapsible={false}
        grow
        rightSlot={
          <button
            type="button"
            onClick={() => workspaceRefreshRef.current?.()}
            className="inline-flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground/70 transition-colors hover:bg-accent hover:text-foreground"
            aria-label="Refresh workspace"
            title="Refresh workspace"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        }
      >
        <div className="min-h-0 flex-1 overflow-y-auto">
          <SidebarWorkspaceTree
            isExpanded
            workspaceVersion={workspaceVersion}
            optimisticRows={optimisticRows}
            onReconciled={onReconciled}
            onDelete={onDeleteWorkspaceFile}
            onRegisterRefresh={registerWorkspaceRefresh}
          />
        </div>
      </SidebarSection>

      {deleteTarget && (
        <DeleteProjectDialog
          name={deleteTarget.name}
          chatCount={deleteTarget.chatIds.length}
          onConfirm={handleConfirmDelete}
          onClose={() => setDeleteTarget(null)}
        />
      )}
    </nav>
  );
}

export const ChatListSidebar = memo(ChatListSidebarInner);

function SidebarSkeleton() {
  return (
    <div className="flex flex-col gap-1.5 px-1">
      {Array.from({ length: 4 }).map((_, i) => (
        <div
          key={i}
          className="h-9 motion-safe:animate-pulse rounded-md bg-muted/40"
        />
      ))}
    </div>
  );
}

interface DeleteAllChatsButtonProps {
  count: number;
  onAfterDelete: () => void;
}

function DeleteAllChatsButton({
  count,
  onAfterDelete,
}: DeleteAllChatsButtonProps) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleConfirm = async () => {
    setBusy(true);
    setError(null);
    try {
      await deleteAllLhaSessions();
      setOpen(false);
      onAfterDelete();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-destructive/15 hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-label="Delete all chats"
        title="Delete all chats"
      >
        <Trash2 className="h-3 w-3" />
      </button>
      {open && (
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete all chats?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently deletes {count} chat
              {count === 1 ? "" : "s"} and can&rsquo;t be undone. Workspace
              files in <span className="font-mono">/workspace</span> are not
              affected.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {error && (
            <div className="rounded border border-destructive/40 bg-destructive/10 px-2 py-1 text-[11px] text-destructive">
              {error}
            </div>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={busy}
              onClick={(e) => {
                e.preventDefault();
                void handleConfirm();
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {busy ? "Deleting…" : `Delete ${count}`}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      )}
    </AlertDialog>
  );
}
