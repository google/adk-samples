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

export type FeedbackSentiment = "up" | "down";

export async function submitFeedback(
  text: string,
  sessionId?: string,
  sentiment?: FeedbackSentiment,
  includeContext?: boolean,
): Promise<void> {
  const body: {
    text: string;
    session_id?: string;
    sentiment?: FeedbackSentiment;
    include_context?: boolean;
  } = { text };
  if (sessionId) body.session_id = sessionId;
  if (sentiment) body.sentiment = sentiment;
  if (includeContext) body.include_context = true;
  const r = await fetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`submitFeedback ${r.status}`);
}
