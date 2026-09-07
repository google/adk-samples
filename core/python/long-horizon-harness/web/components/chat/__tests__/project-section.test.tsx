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
import { render, screen, fireEvent } from "@testing-library/react";
import { ProjectSection } from "@/components/chat/project-section";

describe("ProjectSection", () => {
  it("renders name + count + children and toggles", () => {
    const onToggle = vi.fn();
    render(
      <ProjectSection name="projA" count={3} collapsed={false} onToggle={onToggle}>
        <div>child-row</div>
      </ProjectSection>,
    );
    expect(screen.getByText("projA")).toBeTruthy();
    expect(screen.getByText("3")).toBeTruthy();
    expect(screen.getByText("child-row")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /projA/i }));
    expect(onToggle).toHaveBeenCalledOnce();
  });

  it("hides children when collapsed", () => {
    render(
      <ProjectSection name="p" count={0} collapsed onToggle={() => {}}>
        <div>hidden-row</div>
      </ProjectSection>,
    );
    expect(screen.queryByText("hidden-row")).toBeNull();
  });

  it("fires onNewChat", () => {
    const onNewChat = vi.fn();
    render(
      <ProjectSection
        name="p"
        count={0}
        collapsed={false}
        onToggle={() => {}}
        onNewChat={onNewChat}
      >
        <div />
      </ProjectSection>,
    );
    fireEvent.click(screen.getByRole("button", { name: /new chat in p/i }));
    expect(onNewChat).toHaveBeenCalledOnce();
  });

  it("hides the count badge when count is undefined (lazy group)", () => {
    render(
      <ProjectSection name="Scheduled" collapsed onToggle={() => {}}>
        <div />
      </ProjectSection>,
    );
    // The only text is the name; no numeric badge.
    expect(screen.getByText("Scheduled")).toBeTruthy();
    expect(screen.queryByText(/^\d+$/)).toBeNull();
  });

  it("renders a leading icon when provided", () => {
    render(
      <ProjectSection
        name="Scheduled"
        collapsed
        onToggle={() => {}}
        icon={<svg data-testid="sched-icon" />}
      >
        <div />
      </ProjectSection>,
    );
    expect(screen.getByTestId("sched-icon")).toBeTruthy();
  });

  it("omits new-chat/delete controls for the General bucket", () => {
    render(
      <ProjectSection name="General" count={2} collapsed={false} onToggle={() => {}}>
        <div />
      </ProjectSection>,
    );
    expect(screen.queryByRole("button", { name: /new chat/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /delete project/i })).toBeNull();
  });
});
