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


import { useState } from "react";
import { Lightbulb, ThumbsDown, ThumbsUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { submitFeedback, type FeedbackSentiment } from "@/lib/feedback";

interface FeedbackPopoverProps {
  sessionId?: string;
}

export function FeedbackPopover({ sessionId }: FeedbackPopoverProps) {
  const [text, setText] = useState("");
  const [sentiment, setSentiment] = useState<FeedbackSentiment | null>(null);
  const [includeContext, setIncludeContext] = useState(false);
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState(false);

  const canSend = (text.trim().length > 0 || sentiment !== null) && !sending;

  function toggle(value: FeedbackSentiment) {
    setSentiment((prev) => (prev === value ? null : value));
  }

  async function send() {
    if (!canSend) return;
    setSending(true);
    setError(false);
    try {
      await submitFeedback(
        text.trim(),
        sessionId,
        sentiment ?? undefined,
        includeContext,
      );
      setSent(true);
      setText("");
      setSentiment(null);
      setIncludeContext(false);
    } catch {
      setError(true);
    } finally {
      setSending(false);
    }
  }

  function onOpenChange(open: boolean) {
    if (!open) {
      setSent(false);
      setError(false);
    }
  }

  return (
    <Popover onOpenChange={onOpenChange}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          aria-label="Send feedback"
          title="Have a suggestion?"
        >
          <Lightbulb className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72">
        {sent ? (
          <p className="text-sm text-muted-foreground">Thanks for the feedback!</p>
        ) : (
          <div className="flex flex-col gap-2">
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                size="icon"
                aria-label="Thumbs up"
                aria-pressed={sentiment === "up"}
                className={cn(
                  "h-8 w-8",
                  sentiment === "up" &&
                    "border-emerald-500/60 bg-emerald-500/10 text-emerald-600",
                )}
                onClick={() => toggle("up")}
              >
                <ThumbsUp className="h-4 w-4" />
              </Button>
              <Button
                type="button"
                variant="outline"
                size="icon"
                aria-label="Thumbs down"
                aria-pressed={sentiment === "down"}
                className={cn(
                  "h-8 w-8",
                  sentiment === "down" &&
                    "border-destructive/60 bg-destructive/10 text-destructive",
                )}
                onClick={() => toggle("down")}
              >
                <ThumbsDown className="h-4 w-4" />
              </Button>
            </div>
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Have a suggestion?"
              aria-label="Feedback"
            />
            <label className="flex cursor-pointer items-start gap-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                checked={includeContext}
                onChange={(e) => setIncludeContext(e.target.checked)}
                className="mt-0.5 h-3.5 w-3.5 shrink-0 accent-primary"
              />
              <span>
                Include a redacted summary of this conversation to help your
                team debug.
              </span>
            </label>
            <p className="text-[10px] leading-snug text-muted-foreground/70">
              Saved to this deployment&rsquo;s own logs (Cloud&nbsp;Logging) —
              nothing sent outside your project.
            </p>
            {error && (
              <p className="text-xs text-destructive">
                Couldn&apos;t send — please try again.
              </p>
            )}
            <Button
              type="button"
              size="sm"
              className="self-end"
              disabled={!canSend}
              onClick={send}
            >
              Send
            </Button>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
