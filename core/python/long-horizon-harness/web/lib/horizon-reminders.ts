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


import { useQuery, useQueryClient } from "@tanstack/react-query";
import { qk } from "./query-keys";

export interface HorizonReminder {
  id: string;
  message: string;
  fire_at: string;
  recurrence: "daily" | "weekly" | "hourly" | null;
  channel: string;
}

export interface HorizonRemindersResponse {
  reminders: HorizonReminder[];
}

const KEY = "/lha/reminders";

const fetcher = async (): Promise<HorizonRemindersResponse> => {
  const r = await fetch(KEY, { cache: "no-store" });
  if (!r.ok) throw new Error(`horizon-reminders ${r.status}`);
  return (await r.json()) as HorizonRemindersResponse;
};

export interface UseRemindersResult {
  data: HorizonRemindersResponse | undefined;
  error: Error | undefined;
  isLoading: boolean;
  refresh: () => Promise<unknown>;
}

export function useReminders(): UseRemindersResult {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: qk.reminders(),
    queryFn: fetcher,
    refetchOnWindowFocus: true,
    staleTime: 10_000,
  });
  return {
    data: q.data,
    error: q.error ?? undefined,
    isLoading: q.isLoading,
    refresh: () => qc.invalidateQueries({ queryKey: qk.reminders() }),
  };
}

export async function deleteReminder(id: string): Promise<void> {
  const r = await fetch(`/lha/reminders/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`deleteReminder ${r.status}`);
}
