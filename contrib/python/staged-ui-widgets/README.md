# Staged UI widgets

A shopping agent that answers with real interface — product cards, a comparison
table, a delivery timeline, a spend chart — rendered as [A2UI](https://a2ui.org)
next to its reply.

The interesting part is not that it renders widgets. ADK will let a tool call
`render_ui_widget` the moment it has something to show, and for a single widget
that is the right amount of machinery. This recipe is about what happens after
that: a second widget, a parallel tool call, a shopper who says "bring those
back up" three turns later. Each of those breaks inline rendering in a way that
does not raise an exception, and the fix is the same each time — separate
*deciding what to show* from *sending it*.

So no tool here renders. Every tool writes a payload into session state and
returns a short summary; one `after_agent_callback` at the end of the turn
converts what was staged and emits it. Grep the tools package for
`render_ui_widget` and you get nothing — the only call site is
[app/staging/lifecycle.py](app/staging/lifecycle.py).

## What that split buys

**A widget lifecycle independent of tool calls.** This is the one with no
inline equivalent at all. Because the payload outlives the turn that built it:

- `show_again` re-emits a carousel with no recomputation — it flips two flags
  and the flush does the rest. Without it, "show me those shoes again" is
  answered by re-running the search, which returns *different* shoes once the
  shopper has edited their profile.
- `update_shopper_preference` re-ranks and re-stages the cards on screen in a
  turn where the ranking tool is never called, using the query stored
  alongside the payload.
- `compare_picks([])` compares what the shopper is looking at. They said
  "which of those is better" and named nothing; the ids are in the register.
- A re-rank that comes out byte-identical is *suppressed* rather than resent,
  so the reply cannot announce an update the shopper is unable to see.

**Deterministic emission order.** Inline rendering emits in whatever order the
model happened to call the tools. The flush walks a declared tuple
([app/staging/spec.py](app/staging/spec.py)), so a turn that produces both a
carousel and a chart always emits them in the same sequence. Verified live: a
model that called the spend tool first still got `['ui-picks', 'ui-spend']`.

**Duplicate-id safety under parallel tool calls.** `render_ui_widget` rejects a
duplicate widget id, but each function call gets its own `ToolContext`
(`flows/llm_flows/functions.py:1228`), so the check only ever sees one call's
widgets — and `merge_parallel_function_response_events` concatenates the lists
untouched. Flushing once, from one context, is where that check actually bites.

**An answer to "why is my widget missing".** The flush returns one outcome per
declared widget, emitted or not, with the gate that stopped it: `not staged
this turn`, `already emitted`, `suppressed for this turn`, `register empty`,
`converter produced no components`. A silent skip is a bad answer to the
question this design gets asked most.

## The trap worth reading twice

An `after_agent_callback` that renders widgets and writes nothing produces no
event, and the widgets are discarded with no error anywhere.

`render_ui_widget` appends to the event's *actions*, not to state. ADK creates
an event for a callback only when it returned content **or** state changed
(`agents/base_agent.py:564-582`):

```python
if after_agent_callback_content:
    return Event(..., actions=callback_context._event_actions)
if callback_context.state.has_delta():
    return Event(..., actions=callback_context._event_actions)
return None
```

So a render-only callback returns `None`, no event is created, and the turn
looks exactly like one in which nothing was staged. The emitted flag this
recipe writes per widget is both the dedupe record and the state delta that
forces the event out.
[tests/unit/test_event_delivery.py](tests/unit/test_event_delivery.py) pins it
down with a real `Runner`, because the failure is silent and a refactor could
reintroduce it without breaking anything visible.

## Why the model never writes UI

The model chooses tools and writes prose. It never emits component JSON, for
three reasons that all showed up in testing:

1. **It cannot be validated into correctness.** A2UI is a strict schema —
   parent-before-child ordering, no orphans, a fixed component vocabulary, a
   fixed icon set. Every payload here goes through
   [app/render/converters.py](app/render/converters.py) and is checked against
   the published schema in the test suite. Malformed UI is a code bug, caught
   by `pytest`, not a bad generation caught by a user.
2. **The chart and the sentence cannot disagree.** The headline average and the
   plotted series come from one list. A model narrating a chart it did not draw
   is a model with two chances to round differently.
3. **Tokens.** A product tile is an SVG data URI. It belongs in the payload,
   not in a tool result the model has to read.

What the model *does* get is every fact it might need to reason about — ids,
prices, and the reason chips printed on each card. An early version withheld
the chips, on the theory that a thinner result would stop the model reciting
the card. A live run answered "why did you recommend that one?" by inventing a
rationale from a different tool's output instead. Not duplicating the widget is
the instruction's job ([app/prompt.py](app/prompt.py)); starving the model of
facts only buys a fluent guess.

## See it without credentials

The staging layer, the converters, and the gates need no model and no network:

```bash
uv run python -m app.walkthrough          # one line per turn
uv run python -m app.walkthrough --a2ui   # plus the A2UI messages
```

Eight turns of a shopping conversation with the model's tool choices scripted.
Turn 5 is the suppressed no-op refresh, turn 6 revives a carousel built four
turns earlier, and turn 8 stages nothing — printing `state delta: False`, which
is the trap above made visible.

## Layout

| Path | What lives there |
| ---- | ---------------- |
| [app/staging/](app/staging/) | `spec` declares the widgets, `state` is the API tools call, `lifecycle` is the flush and its gates |
| [app/render/](app/render/) | Payload → A2UI. `components` builds nodes, `converters` maps each semantic type, `registry` dispatches |
| [app/tools/](app/tools/) | The six tools. All stage; none render |
| [app/store.py](app/store.py) | Catalog, orders, spend history, loaded from `app/data/*.json` |
| [app/profile.py](app/profile.py) | The shopper profile: `user:`-scoped, coerced and validated on write |
| [app/ranking.py](app/ranking.py) | Scores products against the profile and produces the reason chips |
| [app/walkthrough.py](app/walkthrough.py) | The credential-free demo above |

Three state scopes carry the whole mechanism, and the difference matters:
`user:shopper_profile` survives across sessions; `ui:register:*` and
`ui:emitted:*` are session-scoped, which is what makes revival possible;
`temp:ui:dirty:*` and `temp:ui:suppress:*` are per-invocation, so a stale flag
cannot emit a widget nobody asked for next turn.

## Requirements

- **uv**: Python package manager — [install](https://docs.astral.sh/uv/getting-started/installation/)
- A Gemini API key, or a Google Cloud project with Vertex AI enabled
- To *see* the widgets rendered, an A2UI-capable host. Everything else —
  including the walkthrough and the full test suite — runs without one.

## Setup

> All commands run from the recipe root (`contrib/python/staged-ui-widgets/`).

```bash
uv sync
cp .env.example .env    # then set GEMINI_API_KEY, or the Vertex AI variables
```

## Run

```bash
uv run adk run app                                   # interactive CLI
uv run uvicorn app.fast_api_app:app --reload         # local web server
```

Things worth trying, in this order — each one exercises a claim above:

1. `I need new trail shoes` — a carousel.
2. `which of those is the best value?` — a comparison of items the shopper
   never named. Ask *why* it carries the flag; the answer is rating per dollar,
   which is a stated metric, not a judgement.
3. `actually I only buy Fellstone now` — the cards re-rank in a turn where the
   ranking tool is not called.
4. `my shoe size hasn't changed` — a refresh that changes nothing, so nothing
   is resent and the reply says so.
5. `bring those shoe cards back up` — revived from the register, not searched
   again.
6. `where's order 9999?` — no widget, and the reply carries the real order ids.

## Running Tests

```bash
uv run pytest              # unit + runnability
uv run pytest tests/unit   # unit only
uv run pytest tests/integration   # needs live credentials
```

The unit suite needs no credentials and no network: it validates real A2UI
against the published schema, exercises every gate, and includes a
`Runner`-level test that a staged widget actually reaches the client.

## Commands

| Command | Description |
| ------- | ----------- |
| `uv run adk run app` | Run the agent in interactive CLI mode |
| `uv run uvicorn app.fast_api_app:app --reload` | Start the local FastAPI development server |
| `uv run python -m app.walkthrough` | Print a scripted conversation's staging outcomes and A2UI |
| `uv run pytest` | Run all test suites |
