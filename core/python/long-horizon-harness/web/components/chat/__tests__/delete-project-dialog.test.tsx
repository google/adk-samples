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
import { DeleteProjectDialog } from "@/components/chat/delete-project-dialog";

describe("DeleteProjectDialog", () => {
  it("shows the name + chat count and confirms", async () => {
    const onConfirm = vi.fn().mockResolvedValue(undefined);
    const onClose = vi.fn();
    render(
      <DeleteProjectDialog
        name="Foo"
        chatCount={2}
        onConfirm={onConfirm}
        onClose={onClose}
      />,
    );

    expect(screen.getByText(/delete project “Foo”/i)).toBeTruthy();
    expect(screen.getByText(/2 chats/i)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /^delete$/i }));
    await waitFor(() => expect(onConfirm).toHaveBeenCalledOnce());
  });

  it("closes on cancel without confirming", () => {
    const onConfirm = vi.fn().mockResolvedValue(undefined);
    const onClose = vi.fn();
    render(
      <DeleteProjectDialog
        name="Foo"
        chatCount={1}
        onConfirm={onConfirm}
        onClose={onClose}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onConfirm).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });
});
