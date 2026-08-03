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


import { useCallback, useEffect, useRef, useState } from "react";

export interface UseMessageQueueArgs {
  busy: boolean;
  // Sends the combined queued text as a single new turn when the current
  // turn ends. Pass the raw `send` from useChatStream — NOT the enqueueing
  // wrapper — so flushing never re-enters the queue.
  send: (text: string) => void;
}

export interface UseMessageQueueResult {
  queue: string[];
  enqueue: (text: string) => void;
  removeQueued: (index: number) => void;
  clearQueue: () => void;
}

export function useMessageQueue({
  busy,
  send,
}: UseMessageQueueArgs): UseMessageQueueResult {
  const [queue, setQueue] = useState<string[]>([]);
  const prevBusyRef = useRef(busy);
  // The flush effect depends only on `busy`; read the live queue/send through
  // refs so an enqueue mid-turn doesn't re-run it and a stale closure doesn't
  // flush an out-of-date queue.
  const queueRef = useRef(queue);
  const sendRef = useRef(send);
  useEffect(() => {
    queueRef.current = queue;
  }, [queue]);
  useEffect(() => {
    sendRef.current = send;
  }, [send]);

  const enqueue = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    setQueue((q) => [...q, trimmed]);
  }, []);

  const removeQueued = useCallback((index: number) => {
    setQueue((q) => q.filter((_, i) => i !== index));
  }, []);

  const clearQueue = useCallback(() => setQueue([]), []);

  // Flush on the natural busy -> idle transition. Stop empties the queue
  // first (chat-shell drains it into the composer), so a stopped turn lands
  // here with nothing to flush.
  useEffect(() => {
    const wasBusy = prevBusyRef.current;
    prevBusyRef.current = busy;
    if (wasBusy && !busy && queueRef.current.length > 0) {
      const text = queueRef.current.join("\n\n");
      setQueue([]);
      sendRef.current(text);
    }
  }, [busy]);

  return { queue, enqueue, removeQueued, clearQueue };
}
