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
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { NewConversationMenu } from "@/components/chat/new-conversation-menu";

function setup() {
  const onGeneral = vi.fn();
  const onPickProject = vi.fn();
  const onCreateProject = vi.fn().mockResolvedValue(undefined);
  render(
    <NewConversationMenu
      projects={["projA", "projB"]}
      onGeneral={onGeneral}
      onPickProject={onPickProject}
      onCreateProject={onCreateProject}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: /new conversation/i }));
  return { onGeneral, onPickProject, onCreateProject };
}

describe("NewConversationMenu", () => {
  it("starts a general chat", () => {
    const { onGeneral } = setup();
    fireEvent.click(screen.getByRole("button", { name: /general chat/i }));
    expect(onGeneral).toHaveBeenCalledOnce();
  });

  it("picks an existing project", () => {
    const { onPickProject } = setup();
    fireEvent.click(screen.getByRole("button", { name: /projA/i }));
    expect(onPickProject).toHaveBeenCalledWith("projA");
  });

  it("creates a project inline and starts a chat in it", async () => {
    const { onCreateProject } = setup();
    fireEvent.click(screen.getByRole("button", { name: /new project/i }));
    const input = screen.getByLabelText("Project name");
    fireEvent.change(input, { target: { value: "projX" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(onCreateProject).toHaveBeenCalledWith("projX"));
  });
});
