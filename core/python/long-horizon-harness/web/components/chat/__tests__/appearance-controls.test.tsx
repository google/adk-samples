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
import userEvent from "@testing-library/user-event";
import { AppearanceControls } from "@/components/chat/appearance-controls";

function setup(overrides = {}) {
  const props = {
    theme: "dark" as const,
    onToggleTheme: vi.fn(),
    skin: "default" as const,
    onSkinChange: vi.fn(),
    fontSize: "default" as const,
    onFontSizeChange: vi.fn(),
    ...overrides,
  };
  render(<AppearanceControls {...props} />);
  return props;
}

describe("AppearanceControls", () => {
  it("calls onSkinChange when a skin swatch is clicked", async () => {
    const props = setup();
    await userEvent.click(screen.getByRole("button", { name: /ocean accent/i }));
    expect(props.onSkinChange).toHaveBeenCalledWith("ocean");
  });

  it("marks the active skin as pressed", () => {
    setup({ skin: "moss" });
    expect(screen.getByRole("button", { name: /moss accent/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: /ocean accent/i })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("calls onFontSizeChange when a size option is clicked", async () => {
    const props = setup();
    await userEvent.click(screen.getByRole("button", { name: "Large text size" }));
    expect(props.onFontSizeChange).toHaveBeenCalledWith("large");
  });

  it("calls onToggleTheme when the theme control is clicked", async () => {
    const props = setup({ theme: "dark" });
    await userEvent.click(screen.getByRole("button", { name: /light theme/i }));
    expect(props.onToggleTheme).toHaveBeenCalled();
  });
});
