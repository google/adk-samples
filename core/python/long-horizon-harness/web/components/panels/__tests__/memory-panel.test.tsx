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
import { render, screen, within, fireEvent } from "@testing-library/react";

import type { HorizonMemoriesResponse } from "@/lib/horizon-memories";

const mockUseLhaMemories = vi.fn();

vi.mock("@/lib/horizon-memories", () => ({
  useLhaMemories: (limit: number) => mockUseLhaMemories(limit),
}));

vi.mock("@/lib/now-context", () => ({
  useNow: () => 1_000_000,
  NowProvider: ({ children }: { children: React.ReactNode }) => children,
}));

import { MemoryPanel } from "@/components/panels/memory-panel";

function fixture(overrides: Partial<HorizonMemoriesResponse> = {}): HorizonMemoriesResponse {
  return {
    counts: { user: 3, agent: 2, user_profile: 1 },
    entries: [
      { scope: "user", content: "prefers espresso", ts_ms: 999_000, session_id: "s1" },
      { scope: "agent", content: "retries flaky", ts_ms: 998_000, session_id: "s1" },
    ],
    has_more: false,
    ...overrides,
  };
}

describe("MemoryPanel", () => {
  it("renders cross-session counts in the pills", () => {
    mockUseLhaMemories.mockReturnValue({
      data: fixture(),
      isLoading: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<MemoryPanel contextId="ctx" />);

    // Pills carry a tooltip so we can disambiguate from the row labels.
    expect(
      screen.getByTitle(/across all your sessions/i),
    ).toHaveTextContent("3");
    expect(
      screen.getByTitle(/the agent decided to remember/i),
    ).toHaveTextContent("2");
    expect(
      screen.getByTitle(/long-term profile facts/i),
    ).toHaveTextContent("1");
  });

  it("renders the entries from the hook", () => {
    mockUseLhaMemories.mockReturnValue({
      data: fixture(),
      isLoading: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<MemoryPanel contextId="ctx" />);

    expect(screen.getByText("prefers espresso")).toBeInTheDocument();
    expect(screen.getByText("retries flaky")).toBeInTheDocument();
  });

  it("shows Load more when has_more is true and triggers a wider fetch", () => {
    mockUseLhaMemories.mockReturnValue({
      data: fixture({ has_more: true }),
      isLoading: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<MemoryPanel contextId="ctx" />);

    const loadMore = screen.getByRole("button", { name: /load more/i });
    expect(loadMore).toBeInTheDocument();

    // Reset the mock so we can capture the next call.
    mockUseLhaMemories.mockClear();
    mockUseLhaMemories.mockReturnValue({
      data: fixture({ has_more: false }),
      isLoading: false,
      error: undefined,
      refresh: vi.fn(),
    });
    fireEvent.click(loadMore);
    const limitsRequested = mockUseLhaMemories.mock.calls.map((c) => c[0]);
    // First render after click should ask for a bigger page than the default 5.
    expect(Math.max(...limitsRequested)).toBeGreaterThan(5);
  });

  it("hides Load more when has_more is false", () => {
    mockUseLhaMemories.mockReturnValue({
      data: fixture({ has_more: false }),
      isLoading: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<MemoryPanel contextId="ctx" />);
    expect(screen.queryByRole("button", { name: /load more/i })).toBeNull();
  });

  it("shows the empty state when no entries and not loading", () => {
    mockUseLhaMemories.mockReturnValue({
      data: { counts: { user: 0, agent: 0, user_profile: 0 }, entries: [], has_more: false },
      isLoading: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<MemoryPanel contextId="ctx" />);
    expect(
      screen.getByText(/asking it to remember a fact about you/i),
    ).toBeInTheDocument();
  });

  it("does not flash the empty state while the first fetch is pending", () => {
    mockUseLhaMemories.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<MemoryPanel contextId="ctx" />);
    expect(
      screen.queryByText(/asking it to remember a fact about you/i),
    ).toBeNull();
  });
});
