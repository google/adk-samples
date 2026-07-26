---
name: self-report
description: Use when you hit a problem with Horizon *itself* worth telling the maintainers — a tool that keeps erroring, a guardrail that misfired on a benign action, a capability/tool you needed but don't have, or an instruction that was confusing or self-contradictory. Files high-signal feedback via the report_to_maintainers tool, which always asks the user before sending.
---

# Self-report to maintainers

You can send feedback about Horizon itself to the people who build it, using the
`report_to_maintainers` tool. You are often the best-placed reporter: you know
exactly which tool failed, what you expected, and what got in your way.

## When to use this skill

Reach for `report_to_maintainers` when, during normal work, you notice
something wrong with *the agent or its tooling* (not with the user's own code):

- A tool errors repeatedly on a reasonable call (≈3+ times) and it looks like a
  defect, not your mistake → `category: "bug"`.
- A guardrail or egress block stops a clearly benign action → `category:
  "guardrail_misfire"`.
- You needed a tool/capability that doesn't exist to finish a normal task →
  `category: "capability_gap"`.
- An instruction in your guidance is contradictory or impossible to follow →
  `category: "confusing_instruction"`.
- Anything else maintainers should know → `category: "other"`.

Do **not** use it for problems in the user's project (file those the user's
way), for one-off transient errors you can work around, or to chat.

## How to file a good report

- One distinct issue per report. The tool de-duplicates within a session, so it
  won't send the same `(category, summary)` twice — don't try to re-send.
- `summary`: a tight one-line title. `details`: what you did, what happened, and
  what you expected — repro-style, like a good bug report.
- Keep `include_context` **false** by default. Set it true only when the
  conversation itself is the evidence the maintainer needs. Even then, the tool
  shows the user the draft and asks before anything is sent, and the attached
  summary is redacted.

## Consent is not yours to skip

`report_to_maintainers` always asks the user to confirm before sending. If the
first call returns `status: "awaiting_user_response"`, stop and let the user
decide — do not claim you filed anything. Only a `status: "filed"` result means
it was sent. A `status: "declined"` means the user said no; respect it and move
on. Never try to send feedback through any other channel.
