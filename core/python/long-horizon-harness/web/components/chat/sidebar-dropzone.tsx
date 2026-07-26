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


import {
  type ChangeEvent,
  type DragEvent,
  useCallback,
  useReducer,
  useRef,
  useState,
} from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

const MAX_BYTES = 50 * 1024 * 1024;
const MAX_FILES_PER_DROP = 200;

interface QueuedFile {
  file: File;
  // Forward-slash relative path including the top folder name, e.g.
  // "myproj/src/main.py". Omitted for plain file uploads.
  relpath?: string;
}

// Minimal subset of the (non-standard) WebKit File System entry API we
// touch. The React DOM types don't ship these.
interface WebkitFileEntry {
  isFile: boolean;
  isDirectory: boolean;
  name: string;
  fullPath: string;
  file?: (cb: (f: File) => void, err?: (e: unknown) => void) => void;
  createReader?: () => {
    readEntries: (
      cb: (entries: WebkitFileEntry[]) => void,
      err?: (e: unknown) => void,
    ) => void;
  };
}

async function readFolderEntry(entry: WebkitFileEntry): Promise<QueuedFile[]> {
  if (entry.isFile) {
    const file: File = await new Promise((resolve, reject) => {
      entry.file?.(resolve, reject);
    });
    // fullPath starts with "/" — strip it for the backend's relpath form.
    return [{ file, relpath: entry.fullPath.replace(/^\//, "") }];
  }
  if (!entry.isDirectory || !entry.createReader) return [];
  const reader = entry.createReader();
  const results: QueuedFile[] = [];
  // readEntries returns batches; loop until an empty batch.
  // (Chrome caps each batch at ~100.)
  while (true) {
    const batch: WebkitFileEntry[] = await new Promise((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (batch.length === 0) break;
    for (const child of batch) {
      results.push(...(await readFolderEntry(child)));
    }
  }
  return results;
}

export interface UploadResult {
  name: string;
  size: number;
  path: string;
  replaced: boolean;
}

interface SidebarDropzoneProps {
  /** Fired after each successful upload — caller appends to recent list,
   * pending-prefix list, and the workspace tree's optimistic rows. */
  onUploadComplete: (result: UploadResult) => void;
  /** Fired (debounced 300 ms by caller) after every successful upload —
   * caller bumps a version that the workspace tree refetches on. */
  onWorkspaceChanged: () => void;
}

type FileState = "queued" | "uploading" | "done" | "failed";

interface FileEntry {
  id: string;
  name: string;
  size: number;
  state: FileState;
  error?: string;
}

interface UploadState {
  entries: FileEntry[];
  // Total enqueued in the current "burst" (cleared once everything settles
  // and the user sees no in-flight work). Drives the "N of M uploaded" line.
  burstTotal: number;
  burstCompleted: number;
}

type Action =
  | { type: "queue"; entries: FileEntry[] }
  | { type: "start"; id: string }
  | { type: "done"; id: string }
  | { type: "fail"; id: string; error: string }
  | { type: "clear" };

function reducer(state: UploadState, action: Action): UploadState {
  switch (action.type) {
    case "queue": {
      // Start a fresh burst once the previous one drained.
      const draining =
        state.entries.filter((e) => e.state !== "failed").length === 0;
      const burstTotal = draining
        ? action.entries.length
        : state.burstTotal + action.entries.length;
      const burstCompleted = draining ? 0 : state.burstCompleted;
      return {
        entries: [...state.entries, ...action.entries],
        burstTotal,
        burstCompleted,
      };
    }
    case "start":
      return {
        ...state,
        entries: state.entries.map((e) =>
          e.id === action.id ? { ...e, state: "uploading" } : e,
        ),
      };
    case "done":
      return {
        ...state,
        entries: state.entries.filter((e) => e.id !== action.id),
        burstCompleted: state.burstCompleted + 1,
      };
    case "fail":
      return {
        ...state,
        entries: state.entries.map((e) =>
          e.id === action.id
            ? { ...e, state: "failed", error: action.error }
            : e,
        ),
      };
    case "clear":
      return { entries: [], burstTotal: 0, burstCompleted: 0 };
  }
}

const INITIAL_STATE: UploadState = {
  entries: [],
  burstTotal: 0,
  burstCompleted: 0,
};

export function SidebarDropzone({
  onUploadComplete,
  onWorkspaceChanged,
}: SidebarDropzoneProps) {
  const [hover, setHover] = useState(false);
  const [rejected, setRejected] = useState<string[]>([]);
  const [uploadState, dispatch] = useReducer(reducer, INITIAL_STATE);
  const entries = uploadState.entries;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const dragDepthRef = useRef(0);

  const accept = useCallback(
    async (queued: QueuedFile[]) => {
      setRejected([]);
      if (queued.length === 0) return;
      if (queued.length > MAX_FILES_PER_DROP) {
        setRejected([`max ${MAX_FILES_PER_DROP} files per drop`]);
        return;
      }
      const rejects: string[] = [];
      const accepted: QueuedFile[] = [];
      for (const q of queued) {
        if (q.file.size > MAX_BYTES) {
          rejects.push(
            `${q.relpath ?? q.file.name} (>${MAX_BYTES / (1024 * 1024)}MB)`,
          );
          continue;
        }
        accepted.push(q);
      }
      if (rejects.length) setRejected(rejects);

      const ids = accepted.map((q) => ({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        name: q.relpath ?? q.file.name,
        size: q.file.size,
        state: "queued" as FileState,
      }));
      dispatch({ type: "queue", entries: ids });

      await Promise.allSettled(
        accepted.map(async (q, i) => {
          const id = ids[i].id;
          dispatch({ type: "start", id });
          try {
            const form = new FormData();
            form.append("file", q.file, q.file.name);
            if (q.relpath) form.append("relpath", q.relpath);
            const resp = await fetch("/lha/uploads", {
              method: "POST",
              body: form,
            });
            if (!resp.ok) {
              const detail = await resp.text();
              throw new Error(
                `${resp.status}: ${detail.slice(0, 200) || "upload failed"}`,
              );
            }
            const body = (await resp.json()) as {
              path: string;
              size: number;
              replaced: boolean;
            };
            onUploadComplete({
              name: q.relpath ?? q.file.name,
              size: body.size,
              path: body.path,
              replaced: body.replaced,
            });
            dispatch({ type: "done", id });
            onWorkspaceChanged();
          } catch (err) {
            dispatch({
              type: "fail",
              id,
              error: err instanceof Error ? err.message : String(err),
            });
          }
        }),
      );
    },
    [onUploadComplete, onWorkspaceChanged],
  );

  const onDragEnter = (e: DragEvent<HTMLDivElement>) => {
    if (!hasFileDrag(e)) return;
    e.preventDefault();
    dragDepthRef.current += 1;
    setHover(true);
  };
  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    if (!hasFileDrag(e)) return;
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) setHover(false);
  };
  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    if (!hasFileDrag(e)) return;
    e.preventDefault();
  };
  const onDrop = async (e: DragEvent<HTMLDivElement>) => {
    if (!hasFileDrag(e)) return;
    e.preventDefault();
    e.stopPropagation();
    dragDepthRef.current = 0;
    setHover(false);

    // Prefer DataTransferItem.webkitGetAsEntry so we can detect folder
    // drops and walk their contents. Fall back to dataTransfer.files when
    // the browser doesn't expose the entry API.
    const items = e.dataTransfer.items;
    const collected: QueuedFile[] = [];
    if (items && items.length > 0 && typeof items[0].webkitGetAsEntry === "function") {
      for (let i = 0; i < items.length; i++) {
        if (items[i].kind !== "file") continue;
        const entry = items[i].webkitGetAsEntry() as WebkitFileEntry | null;
        if (!entry) continue;
        collected.push(...(await readFolderEntry(entry)));
      }
    } else {
      for (const f of Array.from(e.dataTransfer.files ?? [])) {
        collected.push({ file: f });
      }
    }
    void accept(collected);
  };

  const onPicked = (e: ChangeEvent<HTMLInputElement>) => {
    const list = e.target.files;
    if (!list) return;
    void accept(Array.from(list).map((f) => ({ file: f })));
    e.target.value = "";
  };

  const onFolderPicked = (e: ChangeEvent<HTMLInputElement>) => {
    const list = e.target.files;
    if (!list) return;
    // webkitRelativePath looks like "myproj/src/main.py" already.
    const queued: QueuedFile[] = Array.from(list).map((f) => ({
      file: f,
      relpath:
        (f as File & { webkitRelativePath?: string }).webkitRelativePath ||
        f.name,
    }));
    void accept(queued);
    e.target.value = "";
  };

  const failed = entries.filter((e) => e.state === "failed");
  const inFlight = entries.filter((e) => e.state !== "failed");

  return (
    <div className="flex flex-col gap-1.5 px-2 py-2">
      <div
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onDrop={onDrop}
        className={cn(
          "flex flex-col items-center justify-center gap-1 rounded-md border border-dashed px-3 py-3 text-center text-[11px] text-muted-foreground transition-colors",
          hover
            ? "border-primary bg-primary/5 text-primary"
            : "border-border/70 hover:border-border",
        )}
      >
        <Upload className="h-3.5 w-3.5" aria-hidden />
        <span className="leading-tight">
          Drop files or folders into{" "}
          <code className="font-mono text-[10px]">/workspace/uploads</code>
        </span>
        <div className="flex items-center gap-2 text-[10px] text-primary">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="underline-offset-2 hover:underline"
          >
            browse files
          </button>
          <span className="text-muted-foreground/40">·</span>
          <button
            type="button"
            onClick={() => folderInputRef.current?.click()}
            className="underline-offset-2 hover:underline"
          >
            browse folder
          </button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          hidden
          onChange={onPicked}
        />
        <input
          ref={folderInputRef}
          type="file"
          hidden
          onChange={onFolderPicked}
          // webkitdirectory is non-standard; React doesn't type it, so we
          // splat via a string-keyed prop. Supported in Chrome/Edge/Safari/Firefox.
          {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
        />
      </div>
      {inFlight.length > 0 && (
        <div className="flex flex-col gap-1 px-1">
          {uploadState.burstTotal > 1 && (
            <div className="flex items-center justify-between text-[10px] text-muted-foreground">
              <span>
                {uploadState.burstCompleted} of {uploadState.burstTotal} uploaded
              </span>
              <span className="font-mono text-muted-foreground/60">
                {Math.round(
                  (uploadState.burstCompleted / uploadState.burstTotal) * 100,
                )}
                %
              </span>
            </div>
          )}
          {uploadState.burstTotal > 1 && (
            <div className="h-0.5 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full bg-primary transition-all"
                style={{
                  width: `${(uploadState.burstCompleted / uploadState.burstTotal) * 100}%`,
                }}
              />
            </div>
          )}
          <ul className="flex flex-col gap-0.5 text-[10px] text-muted-foreground">
            {inFlight.map((e) => (
              <li key={e.id} className="flex items-center gap-1.5">
                <span
                  className={cn(
                    "h-1 w-1 rounded-full",
                    e.state === "uploading"
                      ? "bg-primary motion-safe:animate-pulse"
                      : "bg-muted-foreground/40",
                  )}
                />
                <span className="truncate">{e.name}</span>
                <span className="ml-auto shrink-0 text-muted-foreground/50">
                  {e.state === "uploading" ? "uploading…" : "queued"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {failed.length > 0 && (
        <ul className="flex flex-col gap-0.5 px-1 text-[10px] text-destructive">
          {failed.map((e) => (
            <li key={e.id} className="truncate" title={e.error}>
              ⚠ {e.name} — {e.error}
            </li>
          ))}
        </ul>
      )}
      {rejected.length > 0 && (
        <ul className="flex flex-col gap-0.5 px-1 text-[10px] text-destructive">
          {rejected.map((r, i) => (
            <li key={i} className="truncate">
              ⚠ {r}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function hasFileDrag(e: DragEvent<HTMLElement>): boolean {
  const types = e.dataTransfer?.types;
  if (!types) return false;
  for (let i = 0; i < types.length; i++) {
    if (types[i] === "Files") return true;
  }
  return false;
}
