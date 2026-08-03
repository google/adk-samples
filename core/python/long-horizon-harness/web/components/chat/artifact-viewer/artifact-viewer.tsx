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

import { useEffect, useState } from "react";
import {
  Braces,
  Code,
  Download,
  File as FileIcon,
  FileText,
  Globe,
  Image as ImageIcon,
  Music,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { CopyButton } from "../copy-button";
import { useViewer } from "./viewer-context";
import {
  ArtifactBody,
  dataUrl,
  decodeText,
  pickRenderer,
  supportsPreview,
  type RendererKind,
  type ViewMode,
} from "./renderers";

const KIND_ICON: Record<RendererKind, typeof FileText> = {
  markdown: FileText,
  code: Code,
  image: ImageIcon,
  audio: Music,
  html: Globe,
  json: Braces,
  download: FileIcon,
};

const TEXT_KINDS: ReadonlySet<RendererKind> = new Set([
  "markdown",
  "code",
  "json",
  "html",
]);

export function ArtifactViewer() {
  const { tabs, activeId, activateTab, closeTab } = useViewer();
  const [mode, setMode] = useState<ViewMode>("preview");

  // A freshly-activated tab always opens in its rendered view.
  useEffect(() => {
    setMode("preview");
  }, [activeId]);

  const active = tabs.find((t) => t.id === activeId) ?? null;

  if (!active) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-xs text-muted-foreground">
        No file open. Open a file from the chat to view it here.
      </div>
    );
  }

  const kind = pickRenderer(active.mimeType, active.name);
  const canPreview = supportsPreview(kind);
  const copyable = TEXT_KINDS.has(kind);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div
        role="tablist"
        aria-label="Open files"
        className="flex shrink-0 items-stretch gap-0.5 overflow-x-auto border-b border-border/60 px-1"
      >
        {tabs.map((t) => {
          const Icon = KIND_ICON[pickRenderer(t.mimeType, t.name)];
          const selected = t.id === activeId;
          return (
            <div
              key={t.id}
              className={cn(
                "group/tab flex items-center gap-1.5 border-b-2 py-1.5 pl-2 pr-1 text-xs transition-colors",
                selected
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground",
              )}
            >
              <button
                type="button"
                role="tab"
                aria-selected={selected}
                onClick={() => activateTab(t.id)}
                className="flex max-w-[16ch] items-center gap-1.5 truncate focus-visible:outline-none"
                title={t.name}
              >
                <Icon className="h-3.5 w-3.5 shrink-0" />
                <span className="truncate">{t.name}</span>
              </button>
              <button
                type="button"
                aria-label={`Close ${t.name}`}
                onClick={() => closeTab(t.id)}
                className="rounded p-0.5 text-muted-foreground/60 opacity-0 transition-opacity hover:bg-muted hover:text-foreground group-hover/tab:opacity-100 focus-visible:opacity-100"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          );
        })}
      </div>

      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border/60 px-3 py-1.5">
        <span className="min-w-0 truncate font-mono text-[11px] text-muted-foreground">
          {active.name}
        </span>
        <div className="flex shrink-0 items-center gap-1">
          {canPreview && (
            <div className="flex items-center rounded-md border border-border/60 p-0.5 text-[11px]">
              {(["preview", "raw"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={cn(
                    "rounded px-2 py-0.5 capitalize transition-colors",
                    mode === m
                      ? "bg-muted text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {m}
                </button>
              ))}
            </div>
          )}
          {copyable && <CopyButton getText={() => decodeText(active) ?? ""} />}
          {(() => {
            const href = dataUrl(active);
            return (
              href && (
                <a
                  href={href}
                  download={active.name}
                  aria-label="Download"
                  className="inline-flex h-7 w-7 items-center justify-center rounded-md border bg-card text-muted-foreground shadow-sm transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring max-md:h-11 max-md:w-11"
                >
                  <Download className="h-3 w-3" />
                </a>
              )
            );
          })()}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        <ArtifactBody
          tab={active}
          kind={kind}
          mode={canPreview ? mode : "preview"}
        />
      </div>
    </div>
  );
}
