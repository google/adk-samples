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

import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import type { HorizonStateResponse } from "@/lib/horizon-state";

const mockUseLhaState = vi.fn();

vi.mock("@/lib/horizon-state", async () => {
  const actual =
    await vi.importActual<typeof import("@/lib/horizon-state")>("@/lib/horizon-state");
  return {
    ...actual,
    useLhaState: (
      contextId: string | null,
      options?: { select?: (d: HorizonStateResponse) => unknown },
    ) => mockUseLhaState(contextId, options),
  };
});

import { SkillsPanel } from "@/components/panels/skills-panel";

function mockState(state: Partial<HorizonStateResponse["state"]>) {
  const response = {
    contextId: "ctx",
    exists: true,
    state: {
      bound_skills: [],
      skill_telemetry: {},
      ...state,
    },
  } as HorizonStateResponse;
  mockUseLhaState.mockImplementation((_ctx, options) => ({
    data: options?.select ? options.select(response) : response,
    error: undefined,
    isLoading: false,
  }));
}

describe("SkillsPanel", () => {
  it("lists installed skills before invocation, merging usage counts", () => {
    mockState({
      bound_skills: [
        { name: "policy", description: "rules", source: "builtin" },
        { name: "my-changelog", description: "changelog", source: "user" },
      ],
      skill_telemetry: {
        policy: { views: 2, manages: 0, last_used: "2026-06-01T00:00:00Z" },
      },
    });

    render(<SkillsPanel contextId="ctx" />);

    // Both installed skills render even though my-changelog was never invoked.
    expect(screen.getByText("policy")).toBeInTheDocument();
    expect(screen.getByText("my-changelog")).toBeInTheDocument();

    // policy shows its 2 views; my-changelog has zero usage (no "×" badge).
    expect(screen.getByText("2×")).toBeInTheDocument();
  });

  it("shows a connecting state when there is no context", () => {
    mockState({ bound_skills: [], skill_telemetry: {} });
    render(<SkillsPanel contextId={null} />);
    expect(screen.getByText(/connecting/i)).toBeInTheDocument();
  });
});
