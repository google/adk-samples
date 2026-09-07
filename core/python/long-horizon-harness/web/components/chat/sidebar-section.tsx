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


import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface SidebarSectionProps {
  title: string;
  storageKey: string;
  defaultOpen?: boolean;
  collapsible?: boolean;
  /** Grow to fill the sidebar's remaining vertical space (its body scrolls internally). */
  grow?: boolean;
  rightSlot?: ReactNode;
  /** Extra classes for the header row — e.g. `pr-10` to clear an overlaid close button. */
  headerClassName?: string;
  children: ReactNode;
  onOpenChange?: (open: boolean) => void;
}

export function SidebarSection({
  title,
  storageKey,
  defaultOpen = true,
  collapsible = true,
  grow = false,
  rightSlot,
  headerClassName,
  children,
  onOpenChange,
}: SidebarSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const raw = window.localStorage.getItem(storageKey);
    if (raw === "1") setOpen(true);
    else if (raw === "0") setOpen(false);
  }, [storageKey]);

  const toggle = useCallback(() => {
    if (!collapsible) return;
    const next = !open;
    setOpen(next);
    try {
      window.localStorage.setItem(storageKey, next ? "1" : "0");
    } catch {
      // localStorage unavailable (private mode); state still flips
    }
    onOpenChange?.(next);
  }, [collapsible, open, storageKey, onOpenChange]);

  // Skip the first run so we don't fire before the localStorage-restoration
  // effect has had a chance to settle the real initial value.
  const didMountRef = useRef(false);
  useEffect(() => {
    if (!didMountRef.current) {
      didMountRef.current = true;
      return;
    }
    if (open) onOpenChange?.(true);
  }, [open, onOpenChange]);

  return (
    <section className={cn("flex flex-col", grow && "min-h-0 flex-1")}>
      <div className={cn("flex h-8 shrink-0 items-center gap-1 border-b px-3", headerClassName)}>
        {collapsible ? (
          <button
            type="button"
            onClick={toggle}
            className="flex h-full flex-1 items-center gap-1.5 text-left text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/80 transition-colors hover:text-foreground"
            aria-expanded={open}
          >
            <ChevronRight
              className={cn(
                "h-3 w-3 shrink-0 transition-transform",
                open && "rotate-90",
              )}
            />
            {title}
          </button>
        ) : (
          <span className="flex h-full flex-1 items-center text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/80">
            {title}
          </span>
        )}
        {rightSlot}
      </div>
      {open && (
        <div className={cn("flex flex-col", grow && "min-h-0 flex-1")}>{children}</div>
      )}
    </section>
  );
}
