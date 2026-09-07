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


import { FileIcon } from "lucide-react";
import { formatBytes } from "./input-box";

export interface RecentUpload {
  name: string;
  size: number;
  uploadedAt: number;
}

interface SidebarUploadListProps {
  items: RecentUpload[];
}

export function SidebarUploadList({ items }: SidebarUploadListProps) {
  if (items.length === 0) return null;
  return (
    <>
      <p className="px-3 pt-2 text-[10px] text-muted-foreground/70">
        These files stay in <span className="font-mono">/workspace/uploads</span>{" "}
        between sessions — only this list is per-session.
      </p>
      <ul className="flex flex-col gap-0.5 px-2 pb-2 pt-1">
        {items.map((it) => (
          <li
            key={`${it.name}-${it.uploadedAt}`}
            className="flex items-center gap-1.5 rounded-md px-1.5 py-1 text-xs text-muted-foreground"
          >
            <FileIcon className="h-3 w-3 shrink-0" />
            <span className="min-w-0 flex-1 truncate" title={it.name}>
              {it.name}
            </span>
            <span className="shrink-0 text-[10px] text-muted-foreground/70">
              {formatBytes(it.size)}
            </span>
            <span className="shrink-0 font-mono text-[10px] text-muted-foreground/60">
              {formatRelative(it.uploadedAt)}
            </span>
          </li>
        ))}
      </ul>
    </>
  );
}

function formatRelative(ts: number): string {
  const diff = Date.now() - ts;
  if (diff < 60_000) return "now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h`;
  return `${Math.floor(diff / 86_400_000)}d`;
}
