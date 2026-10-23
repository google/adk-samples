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


import { Moon, Sun } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  FONT_SIZES,
  SKINS,
  type FontSize,
  type Skin,
  type Theme,
} from "@/lib/use-appearance";

const SKIN_LABELS: Record<Skin, string> = {
  default: "Default",
  ocean: "Ocean",
  ember: "Ember",
  moss: "Moss",
};

// Preview swatch color per skin — mirrors the --primary set in globals.css so
// the picker reads as the accent it applies.
const SKIN_SWATCH: Record<Skin, string> = {
  default: "hsl(220 16% 42%)",
  ocean: "hsl(205 90% 48%)",
  ember: "hsl(18 85% 52%)",
  moss: "hsl(152 48% 38%)",
};

const FONT_LABELS: Record<FontSize, string> = {
  small: "Small",
  default: "Default",
  large: "Large",
  xlarge: "Extra large",
};

export function AppearanceControls({
  theme,
  onToggleTheme,
  skin,
  onSkinChange,
  fontSize,
  onFontSizeChange,
}: {
  theme: Theme;
  onToggleTheme: () => void;
  skin: Skin;
  onSkinChange: (skin: Skin) => void;
  fontSize: FontSize;
  onFontSizeChange: (size: FontSize) => void;
}) {
  return (
    <div className="flex w-56 flex-col gap-4 text-sm">
      <section className="flex flex-col gap-2">
        <h3 className="text-label font-medium text-muted-foreground">Theme</h3>
        <button
          type="button"
          onClick={onToggleTheme}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          className="lh-sidebar-hover flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-foreground"
        >
          {theme === "dark" ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
          <span>{theme === "dark" ? "Light" : "Dark"}</span>
        </button>
      </section>

      <section className="flex flex-col gap-2">
        <h3 className="text-label font-medium text-muted-foreground">Accent</h3>
        <div className="flex items-center gap-2">
          {SKINS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => onSkinChange(s)}
              aria-label={`${SKIN_LABELS[s]} accent`}
              aria-pressed={skin === s}
              title={SKIN_LABELS[s]}
              className={cn(
                "h-6 w-6 rounded-full border transition-transform hover:scale-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                skin === s ? "ring-2 ring-ring ring-offset-2" : "ring-0",
              )}
              style={{ backgroundColor: SKIN_SWATCH[s] }}
            />
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-2">
        <h3 className="text-label font-medium text-muted-foreground">Text size</h3>
        <div className="flex items-stretch overflow-hidden rounded-md border">
          {FONT_SIZES.map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => onFontSizeChange(f)}
              aria-label={`${FONT_LABELS[f]} text size`}
              aria-pressed={fontSize === f}
              title={FONT_LABELS[f]}
              className={cn(
                "flex-1 border-r px-2 py-1 last:border-r-0",
                fontSize === f
                  ? "bg-primary text-primary-foreground"
                  : "lh-sidebar-hover text-foreground",
                f === "small" && "text-[11px]",
                f === "default" && "text-[13px]",
                f === "large" && "text-[15px]",
                f === "xlarge" && "text-[17px]",
              )}
            >
              A
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
