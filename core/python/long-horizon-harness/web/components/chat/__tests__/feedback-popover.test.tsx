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

import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { FeedbackPopover } from "@/components/chat/feedback-popover";
import { submitFeedback } from "@/lib/feedback";

vi.mock("@/lib/feedback", () => ({
  submitFeedback: vi.fn(),
}));

const mockSubmit = vi.mocked(submitFeedback);

async function openPopover() {
  await userEvent.click(screen.getByRole("button", { name: /send feedback/i }));
}

describe("FeedbackPopover", () => {
  beforeEach(() => {
    mockSubmit.mockReset();
    mockSubmit.mockResolvedValue(undefined);
  });

  it("submits typed text with the session id", async () => {
    render(<FeedbackPopover sessionId="ctx-123" />);
    await openPopover();
    await userEvent.type(
      screen.getByPlaceholderText(/suggestion/i),
      "please add dark mode",
    );
    await userEvent.click(screen.getByRole("button", { name: /^send$/i }));
    expect(mockSubmit).toHaveBeenCalledWith(
      "please add dark mode",
      "ctx-123",
      undefined,
      false,
    );
  });

  it("submits a thumbs-up with no text", async () => {
    render(<FeedbackPopover sessionId="ctx-123" />);
    await openPopover();
    await userEvent.click(screen.getByRole("button", { name: /thumbs up/i }));
    await userEvent.click(screen.getByRole("button", { name: /^send$/i }));
    expect(mockSubmit).toHaveBeenCalledWith("", "ctx-123", "up", false);
  });

  it("opts into sharing conversation context when checked", async () => {
    render(<FeedbackPopover sessionId="ctx-123" />);
    await openPopover();
    await userEvent.type(screen.getByPlaceholderText(/suggestion/i), "broken");
    await userEvent.click(screen.getByRole("checkbox"));
    await userEvent.click(screen.getByRole("button", { name: /^send$/i }));
    expect(mockSubmit).toHaveBeenCalledWith("broken", "ctx-123", undefined, true);
  });

  it("disables send when there is neither text nor a thumb", async () => {
    render(<FeedbackPopover />);
    await openPopover();
    const send = screen.getByRole("button", { name: /^send$/i });
    expect(send).toBeDisabled();
    await userEvent.type(screen.getByPlaceholderText(/suggestion/i), "   ");
    expect(send).toBeDisabled();
    expect(mockSubmit).not.toHaveBeenCalled();
  });

  it("shows a confirmation after a successful submit", async () => {
    render(<FeedbackPopover />);
    await openPopover();
    await userEvent.type(screen.getByPlaceholderText(/suggestion/i), "nice tool");
    await userEvent.click(screen.getByRole("button", { name: /^send$/i }));
    expect(await screen.findByText(/thanks/i)).toBeInTheDocument();
  });
});
