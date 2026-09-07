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

import { describe, it, expect, vi, afterEach } from "vitest";
import { connectGoogle, disconnectGoogle, isExpired } from "../gcp-connection";

afterEach(() => vi.unstubAllGlobals());

describe("isExpired", () => {
  const nowSec = () => Math.floor(Date.now() / 1000);

  it("treats a missing expiry as not expired", () => {
    expect(isExpired(null)).toBe(false);
    expect(isExpired(undefined)).toBe(false);
  });

  it("is false for a token that still has time left", () => {
    expect(isExpired(nowSec() + 3600)).toBe(false);
  });

  it("is true once the expiry is in the past (the ~0m case)", () => {
    expect(isExpired(nowSec() - 60)).toBe(true);
  });
});

describe("connectGoogle internet-access caution", () => {
  it("aborts without hitting the network when the user cancels the confirm", async () => {
    const fetchSpy = vi.fn();
    const openSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    vi.stubGlobal("open", openSpy);
    vi.stubGlobal("confirm", vi.fn().mockReturnValue(false));

    await connectGoogle("gcp");

    expect(globalThis.confirm).toHaveBeenCalledTimes(1);
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(openSpy).not.toHaveBeenCalled();
  });

  it("warns about the sandbox's internet access before connecting", async () => {
    const confirmSpy = vi.fn().mockReturnValue(false);
    vi.stubGlobal("confirm", confirmSpy);
    vi.stubGlobal("fetch", vi.fn());

    await connectGoogle("workspace", { surfaces: ["gmail"], readonly: true });

    expect(confirmSpy).toHaveBeenCalledTimes(1);
    const message = String(confirmSpy.mock.calls[0][0]).toLowerCase();
    expect(message).toContain("internet");
  });

  it("proceeds to the consent flow once the user accepts", async () => {
    const fetchSpy = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ auth_url: "https://consent.example" })),
      );
    const openSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    vi.stubGlobal("open", openSpy);
    vi.stubGlobal("confirm", vi.fn().mockReturnValue(true));

    await connectGoogle("gcp");

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(String(fetchSpy.mock.calls[0][0])).toContain(
      "/lha/gcp/connect?target=gcp",
    );
    expect(openSpy).toHaveBeenCalledWith(
      "https://consent.example",
      "_blank",
      "noopener,noreferrer",
    );
  });
});

describe("disconnectGoogle", () => {
  it("does not prompt the caution (no new access granted)", async () => {
    const confirmSpy = vi.fn().mockReturnValue(true);
    vi.stubGlobal("confirm", confirmSpy);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}")));

    await disconnectGoogle("workspace");

    expect(confirmSpy).not.toHaveBeenCalled();
  });
});
