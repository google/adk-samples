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
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";
import {
  initialViewerState,
  viewerReducer,
  type ArtifactInput,
  type ArtifactTab,
} from "./viewer-store";

interface ViewerContextValue {
  tabs: ArtifactTab[];
  activeId: string | null;
  openArtifact: (artifact: ArtifactInput) => void;
  closeTab: (id: string) => void;
  activateTab: (id: string) => void;
  clear: () => void;
}

const ViewerContext = createContext<ViewerContextValue | null>(null);

// `resetKey` (the active contextId) scopes open tabs to a conversation — tabs
// clear when the user switches chats.
export function ViewerProvider({
  children,
  resetKey,
  onOpen,
}: {
  children: ReactNode;
  resetKey?: string | null;
  onOpen?: () => void;
}) {
  const [state, dispatch] = useReducer(viewerReducer, initialViewerState);
  const onOpenRef = useRef(onOpen);
  onOpenRef.current = onOpen;

  useEffect(() => {
    dispatch({ type: "clear" });
  }, [resetKey]);

  const value = useMemo<ViewerContextValue>(
    () => ({
      tabs: state.tabs,
      activeId: state.activeId,
      openArtifact: (artifact) => {
        dispatch({ type: "open", artifact });
        onOpenRef.current?.();
      },
      closeTab: (id) => dispatch({ type: "close", id }),
      activateTab: (id) => dispatch({ type: "activate", id }),
      clear: () => dispatch({ type: "clear" }),
    }),
    [state],
  );

  return (
    <ViewerContext.Provider value={value}>{children}</ViewerContext.Provider>
  );
}

export function useViewer(): ViewerContextValue {
  const ctx = useContext(ViewerContext);
  if (!ctx) throw new Error("useViewer must be used within a ViewerProvider");
  return ctx;
}

// Non-throwing variant for components that render outside the provider (e.g.
// message bubbles in tests) — a no-op openArtifact keeps them safe.
export function useViewerOptional(): ViewerContextValue | null {
  return useContext(ViewerContext);
}
