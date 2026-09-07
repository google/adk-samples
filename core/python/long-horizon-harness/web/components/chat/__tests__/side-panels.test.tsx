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

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Activity } from "lucide-react";
import { selectCounts, Section } from "@/components/chat/side-panels";
import type { HorizonStateResponse } from "@/lib/horizon-state";

function response(state: Partial<HorizonStateResponse["state"]>): HorizonStateResponse {
  return {
    contextId: "ctx",
    exists: true,
    state: {
      bound_skills: [],
      skill_telemetry: {},
      tool_call_log: [],
      delegate_history: [],
      ...state,
    },
  } as HorizonStateResponse;
}

describe("selectCounts", () => {
  it("derives the skills badge from the installed catalog, not telemetry", () => {
    const counts = selectCounts(
      response({
        bound_skills: [
          { name: "policy", description: "", source: "builtin" },
          { name: "my-changelog", description: "", source: "user" },
        ],
        // Only one skill was ever invoked — the badge must still count both.
        skill_telemetry: { policy: { views: 1 } },
      }),
    );
    expect(counts.skills).toBe(2);
  });

  it("is zero when no skills are bound", () => {
    expect(selectCounts(response({})).skills).toBe(0);
  });
});

describe("SidePanels Section", () => {
  it("renders an open section's body in a bounded scroll container so sibling headers stay anchored", () => {
    render(
      <Section label="Activity" Icon={Activity} count={50} defaultOpen>
        <div data-testid="body">a long list of tool calls</div>
      </Section>,
    );
    const wrapper = screen.getByTestId("body").parentElement;
    expect(wrapper?.className).toContain("overflow-y-auto");
    expect(wrapper?.className).toMatch(/max-h-/);
  });

  it("hides the body when collapsed but always keeps its header visible", () => {
    render(
      <Section label="Memory" Icon={Activity} count={3}>
        <div data-testid="body">memory items</div>
      </Section>,
    );
    expect(screen.queryByTestId("body")).toBeNull();
    expect(
      screen.getByRole("button", { name: /memory/i }),
    ).toBeInTheDocument();
  });

  it("renders a custom badge instead of the numeric count, even when count is 0", () => {
    render(
      <Section
        label="Auth"
        Icon={Activity}
        count={0}
        badge={<span>Connect</span>}
      >
        <div>body</div>
      </Section>,
    );
    // The "0 ⇒ no badge" rule must not swallow an explicit attention badge.
    expect(screen.getByText("Connect")).toBeInTheDocument();
  });

  it("prefers the custom badge over the numeric count when both are present", () => {
    render(
      <Section
        label="Auth"
        Icon={Activity}
        count={2}
        badge={<span>connected</span>}
      >
        <div>body</div>
      </Section>,
    );
    expect(screen.getByText("connected")).toBeInTheDocument();
    expect(screen.queryByText("2")).toBeNull();
  });
});
