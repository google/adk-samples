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

import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useAppearance, useResolvedTheme } from "@/lib/use-appearance";

// happy-dom doesn't ship a localStorage; install a minimal in-memory one.
beforeAll(() => {
  if (typeof window.localStorage === "undefined") {
    const store = new Map<string, string>();
    const mock: Storage = {
      get length() {
        return store.size;
      },
      clear: () => store.clear(),
      getItem: (k) => (store.has(k) ? (store.get(k) as string) : null),
      key: (i) => Array.from(store.keys())[i] ?? null,
      removeItem: (k) => void store.delete(k),
      setItem: (k, v) => void store.set(k, String(v)),
    };
    Object.defineProperty(window, "localStorage", { value: mock, configurable: true });
  }
});

function resetDom() {
  window.localStorage.clear();
  const root = document.documentElement;
  root.classList.remove("dark");
  delete root.dataset.skin;
  delete root.dataset.fontSize;
  document.head.querySelectorAll("#lha-theme-color").forEach((n) => n.remove());
}

beforeEach(resetDom);
afterEach(resetDom);

describe("useAppearance", () => {
  it("setSkin applies the data-skin attribute and persists it", () => {
    const { result } = renderHook(() => useAppearance());
    act(() => result.current.setSkin("ocean"));
    expect(result.current.skin).toBe("ocean");
    expect(document.documentElement.dataset.skin).toBe("ocean");
    expect(window.localStorage.getItem("lha.skin")).toBe("ocean");
  });

  it("setSkin('default') removes the attribute (no skin styling)", () => {
    const { result } = renderHook(() => useAppearance());
    act(() => result.current.setSkin("ember"));
    act(() => result.current.setSkin("default"));
    expect(result.current.skin).toBe("default");
    expect(document.documentElement.dataset.skin).toBeUndefined();
    expect(window.localStorage.getItem("lha.skin")).toBe("default");
  });

  it("setFontSize applies the data-font-size attribute and persists it", () => {
    const { result } = renderHook(() => useAppearance());
    act(() => result.current.setFontSize("large"));
    expect(result.current.fontSize).toBe("large");
    expect(document.documentElement.dataset.fontSize).toBe("large");
    expect(window.localStorage.getItem("lha.fontSize")).toBe("large");
  });

  it("setFontSize('default') removes the attribute", () => {
    const { result } = renderHook(() => useAppearance());
    act(() => result.current.setFontSize("xlarge"));
    act(() => result.current.setFontSize("default"));
    expect(document.documentElement.dataset.fontSize).toBeUndefined();
    expect(window.localStorage.getItem("lha.fontSize")).toBe("default");
  });

  it("toggleTheme flips the dark class, persists, and updates the theme-color meta", () => {
    const meta = document.createElement("meta");
    meta.id = "lha-theme-color";
    meta.setAttribute("name", "theme-color");
    document.head.appendChild(meta);

    const { result } = renderHook(() => useAppearance());
    const startDark = document.documentElement.classList.contains("dark");

    act(() => result.current.toggleTheme());
    expect(document.documentElement.classList.contains("dark")).toBe(!startDark);
    expect(window.localStorage.getItem("lha.theme")).toBe(startDark ? "light" : "dark");
    // meta tracks the resolved theme with the exact --background hex.
    expect(meta.getAttribute("content")).toBe(startDark ? "#fefdfb" : "#101010");
  });

  it("syncs skin from a cross-tab storage event", () => {
    const { result } = renderHook(() => useAppearance());
    act(() => {
      window.localStorage.setItem("lha.skin", "moss");
      window.dispatchEvent(
        new StorageEvent("storage", { key: "lha.skin", newValue: "moss" }),
      );
    });
    expect(result.current.skin).toBe("moss");
    expect(document.documentElement.dataset.skin).toBe("moss");
  });

  it("syncs font-size from a cross-tab storage event", () => {
    const { result } = renderHook(() => useAppearance());
    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", { key: "lha.fontSize", newValue: "xlarge" }),
      );
    });
    expect(result.current.fontSize).toBe("xlarge");
    expect(document.documentElement.dataset.fontSize).toBe("xlarge");
  });

  it("syncs theme from a cross-tab storage event and treats removal as dark", () => {
    const { result } = renderHook(() => useAppearance());
    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", { key: "lha.theme", newValue: "light" }),
      );
    });
    expect(result.current.theme).toBe("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    // A null newValue (key cleared in another tab) resolves to the dark default.
    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", { key: "lha.theme", newValue: null }),
      );
    });
    expect(result.current.theme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });
});

describe("useResolvedTheme", () => {
  it("reflects the initial dark class on <html>", () => {
    document.documentElement.classList.add("dark");
    const { result } = renderHook(() => useResolvedTheme());
    expect(result.current).toBe("dark");
  });

  // Regression: a separate component toggling the theme in the same tab does
  // not fire a `storage` event, so consumers must track the `dark` class itself
  // rather than their own useState — otherwise code blocks keep the stale theme.
  // Each direction renders fresh (one MutationObserver, one transition) to keep
  // the observer's async delivery deterministic under happy-dom.
  it("reacts to the dark class being added in the same tab", async () => {
    const { result } = renderHook(() => useResolvedTheme());
    expect(result.current).toBe("light");
    act(() => {
      document.documentElement.classList.add("dark");
    });
    await waitFor(() => expect(result.current).toBe("dark"));
  });

  it("reacts to the dark class being removed in the same tab", async () => {
    document.documentElement.classList.add("dark");
    const { result } = renderHook(() => useResolvedTheme());
    expect(result.current).toBe("dark");
    act(() => {
      document.documentElement.classList.remove("dark");
    });
    await waitFor(() => expect(result.current).toBe("light"));
  });
});
