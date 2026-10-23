"use client";
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


import { ExternalLink, KeyRound } from "lucide-react";
import { useLhaState, type HorizonStateResponse } from "@/lib/horizon-state";
import { cn } from "@/lib/utils";

interface AuthCardProps {
  contextId: string | null;
  className?: string;
}

const selectPendingCredential = (d: HorizonStateResponse) =>
  d.state.pending_credential;

export function AuthCard({ contextId, className }: AuthCardProps) {
  const { data: pending } = useLhaState(contextId, {
    select: selectPendingCredential,
  });

  if (!contextId || !pending) return null;

  const authUrl = pending.auth_url;

  return (
    <div
      className={cn(
        "mx-auto flex w-full max-w-2xl flex-col gap-2 rounded-lg border bg-card px-4 py-3 text-sm shadow-sm",
        className,
      )}
    >
      <div className="flex items-center gap-2 text-foreground">
        <KeyRound className="h-4 w-4 text-primary" />
        <span className="font-medium">
          Sign in to {pending.provider} to continue
        </span>
      </div>
      {pending.scopes.length > 0 && (
        <div className="text-[11px] text-muted-foreground">
          Requested scopes:{" "}
          <span className="font-mono">{pending.scopes.join(", ")}</span>
        </div>
      )}
      {authUrl ? (
        <a
          href={authUrl}
          target="_blank"
          rel="noreferrer"
          className="inline-flex w-fit items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        >
          <ExternalLink className="h-3 w-3" />
          Open sign-in window
        </a>
      ) : (
        <div className="text-[11px] text-destructive">
          Auth URL missing — check OAuth client configuration.
        </div>
      )}
      <div className="text-[11px] text-muted-foreground">
        After signing in, send any message to resume the chat.
      </div>
    </div>
  );
}
