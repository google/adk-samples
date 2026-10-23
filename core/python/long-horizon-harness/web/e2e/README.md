# Playwright E2E

End-to-end tests for the chat UI. The A2A backend is mocked at the `fetch`
layer via `page.route()` — no real ADK server is required.

## Setup

One-time browser download (~200 MB Chromium):

```
cd web
npm run test:e2e:install
```

## Run

The dev server on `:3000` is reused if already running, so this won't restart
your live session:

```
cd web
npm run test:e2e           # headless
npm run test:e2e:ui        # interactive UI
```

## Fixtures

A2A event streams live as JSONL files under `e2e/fixtures/streams/`. Each line
is one A2A event (the same shape used in `lib/__fixtures__/events/`). The mock
helper (`e2e/helpers/mock-a2a.ts`) replays them as `text/event-stream`.

To capture a real stream for a new spec, point the browser at the live agent,
inspect the SSE response in DevTools, and paste each event JSON on its own
line.
