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

export type Theme = "light" | "dark";
export type Skin = "default" | "ocean" | "ember" | "moss";
export type FontSize = "small" | "default" | "large" | "xlarge";

export const SKINS: Skin[] = ["default", "ocean", "ember", "moss"];
export const FONT_SIZES: FontSize[] = ["small", "default", "large", "xlarge"];

const THEME_KEY = "lha.theme";
const SKIN_KEY = "lha.skin";
const FONT_KEY = "lha.fontSize";

// Must match --background in app/globals.css (:root vs .dark) so the mobile
// browser chrome blends with the page.
const THEME_COLOR: Record<Theme, string> = {
  light: "#fefdfb",
  dark: "#101010",
};

function setItem(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // localStorage unavailable — the change is session-only.
  }
}

function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
  const meta = document.getElementById("lha-theme-color");
  if (meta) meta.setAttribute("content", THEME_COLOR[theme]);
}

function applyAttr(name: "skin" | "fontSize", value: string, isDefault: boolean) {
  if (isDefault) delete document.documentElement.dataset[name];
  else document.documentElement.dataset[name] = value;
}

function initialTheme(): Theme {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

function initialSkin(): Skin {
  if (typeof document === "undefined") return "default";
  const v = document.documentElement.dataset.skin as Skin | undefined;
  return v && SKINS.includes(v) ? v : "default";
}

function initialFontSize(): FontSize {
  if (typeof document === "undefined") return "default";
  const v = document.documentElement.dataset.fontSize as FontSize | undefined;
  return v && FONT_SIZES.includes(v) ? v : "default";
}

// Reactively tracks the resolved theme by watching the `dark` class on <html>.
// applyTheme() toggles that class for every theme change — toggle button,
// cross-tab storage sync, and the pre-paint boot script alike — so a consumer
// that only needs to react to the theme (not change it) stays in sync no matter
// which component triggered the switch, without sharing useState.
export function useResolvedTheme(): Theme {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  useEffect(() => {
    const el = document.documentElement;
    const sync = () =>
      setTheme(el.classList.contains("dark") ? "dark" : "light");
    sync();
    const observer = new MutationObserver(sync);
    observer.observe(el, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);
  return theme;
}

export function useAppearance(): {
  theme: Theme;
  toggleTheme: () => void;
  skin: Skin;
  setSkin: (skin: Skin) => void;
  fontSize: FontSize;
  setFontSize: (size: FontSize) => void;
} {
  const [theme, setThemeState] = useState<Theme>(initialTheme);
  const [skin, setSkinState] = useState<Skin>(initialSkin);
  const [fontSize, setFontSizeState] = useState<FontSize>(initialFontSize);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === THEME_KEY) {
        const next: Theme = e.newValue === "light" ? "light" : "dark";
        setThemeState(next);
        applyTheme(next);
      } else if (e.key === SKIN_KEY) {
        const next = (
          e.newValue && SKINS.includes(e.newValue as Skin) ? e.newValue : "default"
        ) as Skin;
        setSkinState(next);
        applyAttr("skin", next, next === "default");
      } else if (e.key === FONT_KEY) {
        const next = (
          e.newValue && FONT_SIZES.includes(e.newValue as FontSize)
            ? e.newValue
            : "default"
        ) as FontSize;
        setFontSizeState(next);
        applyAttr("fontSize", next, next === "default");
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const toggleTheme = () => {
    setThemeState((t) => {
      const next: Theme = t === "dark" ? "light" : "dark";
      setItem(THEME_KEY, next);
      applyTheme(next);
      return next;
    });
  };

  const setSkin = (next: Skin) => {
    setSkinState(next);
    setItem(SKIN_KEY, next);
    applyAttr("skin", next, next === "default");
  };

  const setFontSize = (next: FontSize) => {
    setFontSizeState(next);
    setItem(FONT_KEY, next);
    applyAttr("fontSize", next, next === "default");
  };

  return { theme, toggleTheme, skin, setSkin, fontSize, setFontSize };
}
