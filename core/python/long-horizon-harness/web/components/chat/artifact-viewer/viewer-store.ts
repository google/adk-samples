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

export interface ArtifactInput {
  name: string;
  mimeType: string;
  /** base64 body, when the artifact travels inline in the FilePart. */
  bytes?: string;
  /** signed/download URL, when the artifact is fetched. */
  url?: string;
}

export interface ArtifactTab extends ArtifactInput {
  id: string;
}

export interface ViewerState {
  tabs: ArtifactTab[];
  activeId: string | null;
}

export const initialViewerState: ViewerState = { tabs: [], activeId: null };

// Identity is content-shaped so re-opening the same artifact activates the
// existing tab instead of stacking duplicates.
export function tabId(a: ArtifactInput): string {
  return `${a.name}|${a.mimeType}|${a.bytes?.length ?? 0}|${a.url ?? ""}`;
}

export type ViewerAction =
  | { type: "open"; artifact: ArtifactInput }
  | { type: "close"; id: string }
  | { type: "activate"; id: string }
  | { type: "clear" };

export function viewerReducer(
  state: ViewerState,
  action: ViewerAction,
): ViewerState {
  switch (action.type) {
    case "open": {
      const id = tabId(action.artifact);
      const exists = state.tabs.some((t) => t.id === id);
      const tabs = exists
        ? state.tabs
        : [...state.tabs, { id, ...action.artifact }];
      return { tabs, activeId: id };
    }
    case "close": {
      const idx = state.tabs.findIndex((t) => t.id === action.id);
      if (idx === -1) return state;
      const tabs = state.tabs.filter((t) => t.id !== action.id);
      let activeId = state.activeId;
      if (state.activeId === action.id) {
        // Prefer the tab that slid into this slot, else the previous one.
        const neighbor = tabs[idx] ?? tabs[idx - 1] ?? null;
        activeId = neighbor ? neighbor.id : null;
      }
      return { tabs, activeId };
    }
    case "activate":
      return state.tabs.some((t) => t.id === action.id)
        ? { ...state, activeId: action.id }
        : state;
    case "clear":
      return initialViewerState;
  }
}
