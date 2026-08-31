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

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const killProcess = vi.fn(async (_id: string) => {});
const refresh = vi.fn(async () => {});
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let mockData: any;

vi.mock("@/lib/horizon-processes", () => ({
  useProcesses: () => ({
    data: mockData,
    error: undefined,
    isLoading: false,
    refresh,
  }),
  killProcess: (id: string) => killProcess(id),
  hasRunning: () => false,
}));

import { BackgroundPanel } from "@/components/panels/background-panel";

beforeEach(() => {
  killProcess.mockClear();
  refresh.mockClear();
  mockData = {
    processes: [
      {
        session_id: "proc_a",
        command: "npm run dev",
        running: true,
        exit_code: null,
        idle_seconds: 3,
        output_size: 0,
        pid: 1,
        started_at: 0,
      },
    ],
  };
});
afterEach(() => vi.restoreAllMocks());

describe("BackgroundPanel", () => {
  it("renders a running process", () => {
    render(<BackgroundPanel />);
    expect(screen.getByText("npm run dev")).toBeInTheDocument();
  });

  it("kills a process after confirm", async () => {
    render(<BackgroundPanel />);
    await userEvent.click(
      screen.getByRole("button", { name: /kill process: npm run dev/i }),
    );
    await userEvent.click(screen.getByRole("button", { name: /^kill$/i }));
    await waitFor(() => expect(killProcess).toHaveBeenCalledWith("proc_a"));
  });

  it("shows empty state when no processes", () => {
    mockData = { processes: [] };
    render(<BackgroundPanel />);
    expect(screen.getByText(/no background processes/i)).toBeInTheDocument();
  });
});

describe("BackgroundPanel: only live processes", () => {
  it("hides processes that have already exited", async () => {
    mockData = {
      processes: [
        {
          session_id: "live", command: "npm run dev", running: true,
          exit_code: null, idle_seconds: 1, output_size: 0, pid: 1, started_at: 0,
        },
        {
          session_id: "done", command: "uv run python scripts/generate_image.py",
          running: false, exit_code: 1, idle_seconds: 99, output_size: 0,
          pid: 2, started_at: 0,
        },
      ],
    };
    render(<BackgroundPanel />);
    expect(screen.getByText("npm run dev")).toBeInTheDocument();
    // A finished process is not "background work"; it also can't be killed,
    // and the section count already excludes it.
    expect(
      screen.queryByText("uv run python scripts/generate_image.py"),
    ).not.toBeInTheDocument();
  });

  it("shows the empty state when every process has exited", () => {
    mockData = {
      processes: [
        {
          session_id: "done", command: "python3 scripts/test_models.py",
          running: false, exit_code: 0, idle_seconds: 5, output_size: 0,
          pid: 3, started_at: 0,
        },
      ],
    };
    render(<BackgroundPanel />);
    expect(
      screen.queryByText("python3 scripts/test_models.py"),
    ).not.toBeInTheDocument();
  });
});
