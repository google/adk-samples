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

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SidebarWorkspaceTree } from "@/components/chat/sidebar-workspace-tree";

const rootListing = {
  path: "/workspace",
  truncated: false,
  entries: [
    { name: "report.pdf", kind: "file", size: 10, mtime: 1 },
    { name: "notes.txt", kind: "file", size: 20, mtime: 2 },
    { name: "data", kind: "dir", size: 0, mtime: 3 },
  ],
};

const emptyListing = { path: "", truncated: false, entries: [] };

beforeEach(() => {
  global.fetch = vi.fn(async (url: string) => {
    const u = String(url);
    if (u.includes("/lha/workspace/tree")) {
      // Root returns the fixture; auto-fetched subdirs (e.g. /workspace/data)
      // return empty so the listing isn't ambiguously duplicated.
      const isRoot = u.includes(encodeURIComponent("/workspace") + "&") ||
        u.endsWith(encodeURIComponent("/workspace"));
      return {
        ok: true,
        json: async () => (isRoot ? rootListing : emptyListing),
      } as Response;
    }
    return { ok: false, status: 404, text: async () => "" } as Response;
  }) as unknown as typeof fetch;
});

afterEach(() => vi.restoreAllMocks());

const baseProps = {
  isExpanded: true as const,
  workspaceVersion: 0,
  optimisticRows: [],
  onReconciled: () => {},
};

describe("SidebarWorkspaceTree search", () => {
  it("renders a search input and all entries initially", async () => {
    render(<SidebarWorkspaceTree {...baseProps} />);
    expect(
      screen.getByRole("searchbox", { name: /search workspace/i }),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByText("report.pdf")).toBeInTheDocument(),
    );
    expect(screen.getByText("notes.txt")).toBeInTheDocument();
  });

  it("filters visible files as the user types and restores on clear", async () => {
    render(<SidebarWorkspaceTree {...baseProps} />);
    await waitFor(() =>
      expect(screen.getByText("report.pdf")).toBeInTheDocument(),
    );
    const box = screen.getByRole("searchbox", { name: /search workspace/i });

    await userEvent.type(box, "report");
    expect(screen.getByText("report.pdf")).toBeInTheDocument();
    expect(screen.queryByText("notes.txt")).not.toBeInTheDocument();

    await userEvent.clear(box);
    expect(screen.getByText("notes.txt")).toBeInTheDocument();
  });

  it("shows a no-match empty state for a query with no hits", async () => {
    render(<SidebarWorkspaceTree {...baseProps} />);
    await waitFor(() =>
      expect(screen.getByText("report.pdf")).toBeInTheDocument(),
    );
    await userEvent.type(
      screen.getByRole("searchbox", { name: /search workspace/i }),
      "zzz-nope",
    );
    expect(screen.getByText(/no files match/i)).toBeInTheDocument();
  });
});

// Per-path mock: each directory path returns its own listing; any path not
// in the map returns empty so auto-fetched subdirs don't echo root entries.
function installTreeMock(listings: Record<string, TreeListing>) {
  global.fetch = vi.fn(async (url: string) => {
    const u = new URL(String(url), "http://localhost");
    if (u.pathname.endsWith("/lha/workspace/tree")) {
      const path = u.searchParams.get("path") ?? "";
      const listing = listings[path] ?? {
        path,
        truncated: false,
        entries: [],
      };
      return { ok: true, json: async () => listing } as Response;
    }
    return { ok: false, status: 404, text: async () => "" } as Response;
  }) as unknown as typeof fetch;
}

interface TreeListing {
  path: string;
  truncated: boolean;
  entries: Array<{
    name: string;
    kind: "file" | "dir" | "symlink";
    size: number;
    mtime: number;
  }>;
}

describe("SidebarWorkspaceTree refresh", () => {
  it("refetches expanded subdirs (not just root) and bypasses the HTTP cache", async () => {
    const calls: Array<{ url: string; init?: RequestInit }> = [];
    let dataEntries = [{ name: "a.txt", kind: "file" as const, size: 1, mtime: 1 }];
    global.fetch = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url: String(url), init });
      const u = new URL(String(url), "http://localhost");
      if (u.pathname.endsWith("/lha/workspace/tree")) {
        const path = u.searchParams.get("path") ?? "";
        if (path === "/workspace") {
          return {
            ok: true,
            json: async () => ({
              path,
              truncated: false,
              entries: [{ name: "data", kind: "dir", size: 0, mtime: 1 }],
            }),
          } as Response;
        }
        if (path === "/workspace/data") {
          return {
            ok: true,
            json: async () => ({ path, truncated: false, entries: dataEntries }),
          } as Response;
        }
      }
      return { ok: false, status: 404, text: async () => "" } as Response;
    }) as unknown as typeof fetch;

    let refresh: (() => void) | undefined;
    render(
      <SidebarWorkspaceTree
        {...baseProps}
        onRegisterRefresh={(fn) => {
          refresh = fn;
        }}
      />,
    );
    await userEvent.click(await screen.findByRole("button", { name: /^data$/i }));
    await waitFor(() => expect(screen.getByText("a.txt")).toBeInTheDocument());

    // The agent writes a new file into the already-expanded subdir.
    dataEntries = [
      ...dataEntries,
      { name: "b.txt", kind: "file", size: 2, mtime: 2 },
    ];

    // Refresh now lives in the section header — invoke the action the tree
    // registered with its parent.
    await act(async () => {
      refresh?.();
    });

    // Refresh must surface the new file in the expanded subdir, not only root.
    await waitFor(() => expect(screen.getByText("b.txt")).toBeInTheDocument());

    // Every tree fetch opts out of the HTTP cache so refresh returns fresh data.
    const treeCalls = calls.filter((c) => c.url.includes("/lha/workspace/tree"));
    expect(treeCalls.length).toBeGreaterThan(0);
    for (const c of treeCalls) expect(c.init?.cache).toBe("no-store");
  });
});

describe("SidebarWorkspaceTree subtree match + auto-fetch", () => {
  it("keeps a parent dir visible when a manually-expanded child matches", async () => {
    installTreeMock({
      "/workspace": {
        path: "/workspace",
        truncated: false,
        entries: [
          { name: "topfile.txt", kind: "file", size: 1, mtime: 1 },
          { name: "data", kind: "dir", size: 0, mtime: 2 },
        ],
      },
      "/workspace/data": {
        path: "/workspace/data",
        truncated: false,
        entries: [
          { name: "needle.csv", kind: "file", size: 5, mtime: 3 },
          { name: "other.bin", kind: "file", size: 7, mtime: 4 },
        ],
      },
    });

    render(<SidebarWorkspaceTree {...baseProps} />);
    await waitFor(() =>
      expect(screen.getByText("topfile.txt")).toBeInTheDocument(),
    );

    // Manually expand `data` so its listing is fetched into subDirs before
    // we filter; this isolates the subtree-match path from auto-fetch.
    await userEvent.click(screen.getByRole("button", { name: /^data$/i }));
    await waitFor(() =>
      expect(screen.getByText("needle.csv")).toBeInTheDocument(),
    );

    // Query matches only the deep child, never the parent dir name.
    await userEvent.type(
      screen.getByRole("searchbox", { name: /search workspace/i }),
      "needle",
    );

    // Parent dir stays visible because a fetched descendant matches…
    expect(screen.getByText("data")).toBeInTheDocument();
    // …and the matching child is shown (dir is force-expanded while filtering).
    expect(screen.getByText("needle.csv")).toBeInTheDocument();
    // Non-matching siblings are filtered out at both levels.
    expect(screen.queryByText("topfile.txt")).not.toBeInTheDocument();
    expect(screen.queryByText("other.bin")).not.toBeInTheDocument();
  });

  it("auto-fetches a collapsed root dir on first filter and keeps the ancestor visible", async () => {
    installTreeMock({
      "/workspace": {
        path: "/workspace",
        truncated: false,
        entries: [
          { name: "topfile.txt", kind: "file", size: 1, mtime: 1 },
          { name: "vault", kind: "dir", size: 0, mtime: 2 },
        ],
      },
      "/workspace/vault": {
        path: "/workspace/vault",
        truncated: false,
        entries: [{ name: "secret-needle.txt", kind: "file", size: 9, mtime: 3 }],
      },
    });

    render(<SidebarWorkspaceTree {...baseProps} />);
    await waitFor(() =>
      expect(screen.getByText("topfile.txt")).toBeInTheDocument(),
    );

    // `vault` was never manually expanded — entering a filter must auto-fetch
    // it one level deep so its contents become searchable.
    await userEvent.type(
      screen.getByRole("searchbox", { name: /search workspace/i }),
      "needle",
    );

    // After auto-fetch resolves, the ancestor dir is kept visible and the
    // matching descendant surfaces (force-expanded).
    await waitFor(() =>
      expect(screen.getByText("secret-needle.txt")).toBeInTheDocument(),
    );
    expect(screen.getByText("vault")).toBeInTheDocument();
    expect(screen.queryByText("topfile.txt")).not.toBeInTheDocument();

    // Clearing the query restores the unfiltered listing without leaving the
    // auto-fetched dir force-expanded.
    await userEvent.clear(
      screen.getByRole("searchbox", { name: /search workspace/i }),
    );
    expect(screen.getByText("topfile.txt")).toBeInTheDocument();
    expect(screen.queryByText("secret-needle.txt")).not.toBeInTheDocument();
  });

  it("auto-fetch is bounded to one level: a grandchild match stays invisible until its dir is expanded", async () => {
    // Auto-fetch only reaches root-level dirs (one level deep). `outer` gets
    // fetched, but its child dir `inner` does not, so a match living two levels
    // deep is unreachable on first filter — the documented limitation.
    installTreeMock({
      "/workspace": {
        path: "/workspace",
        truncated: false,
        entries: [{ name: "outer", kind: "dir", size: 0, mtime: 1 }],
      },
      "/workspace/outer": {
        path: "/workspace/outer",
        truncated: false,
        entries: [{ name: "inner", kind: "dir", size: 0, mtime: 2 }],
      },
      "/workspace/outer/inner": {
        path: "/workspace/outer/inner",
        truncated: false,
        entries: [{ name: "needle.txt", kind: "file", size: 3, mtime: 3 }],
      },
    });

    render(<SidebarWorkspaceTree {...baseProps} />);
    await waitFor(() => expect(screen.getByText("outer")).toBeInTheDocument());

    await userEvent.type(
      screen.getByRole("searchbox", { name: /search workspace/i }),
      "needle",
    );

    // `outer` is auto-fetched, but `inner`'s own listing is never fetched, so
    // the grandchild match is unreachable: `outer` has no fetched matching
    // descendant and is filtered out entirely, leaving a plain no-match state.
    await waitFor(() =>
      expect(screen.getByText(/no files match/i)).toBeInTheDocument(),
    );
    expect(screen.queryByText("needle.txt")).not.toBeInTheDocument();
    expect(screen.queryByText("inner")).not.toBeInTheDocument();

    // Clearing the query restores the (auto-fetched) root dir unchanged.
    await userEvent.clear(
      screen.getByRole("searchbox", { name: /search workspace/i }),
    );
    expect(screen.getByText("outer")).toBeInTheDocument();
  });
});

describe("SidebarWorkspaceTree hidden files", () => {
  it("hides dotfiles by default and reveals them via the toggle", async () => {
    installTreeMock({
      "/workspace": {
        path: "/workspace",
        truncated: false,
        entries: [
          { name: "report.pdf", kind: "file", size: 1, mtime: 1 },
          { name: ".lha", kind: "dir", size: 0, mtime: 2 },
          { name: ".env", kind: "file", size: 3, mtime: 3 },
        ],
      },
    });

    render(<SidebarWorkspaceTree {...baseProps} />);
    await waitFor(() =>
      expect(screen.getByText("report.pdf")).toBeInTheDocument(),
    );

    // Dotfiles hidden by default.
    expect(screen.queryByText(".lha")).not.toBeInTheDocument();
    expect(screen.queryByText(".env")).not.toBeInTheDocument();

    const toggle = screen.getByRole("checkbox", { name: /show hidden files/i });

    // Toggle on -> dotfiles appear; regular files stay.
    await userEvent.click(toggle);
    expect(screen.getByText(".lha")).toBeInTheDocument();
    expect(screen.getByText(".env")).toBeInTheDocument();
    expect(screen.getByText("report.pdf")).toBeInTheDocument();

    // Toggle off -> hidden again.
    await userEvent.click(toggle);
    expect(screen.queryByText(".lha")).not.toBeInTheDocument();
    expect(screen.getByText("report.pdf")).toBeInTheDocument();
  });
});
