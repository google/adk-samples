# Context budget

What one turn costs, what stops it growing, and how to check. Horizon pays a
fixed prefix on every model call, not every user turn, and one complex turn is
10 to 20 calls, so a char added here is paid 10 to 20 times.

Related: [`architecture.md`](architecture.md) (where the tiers are assembled),
[`memory.md`](memory.md) (compaction and recall), [`configuration.md`](configuration.md)
(the env vars named here).

## The fixed prefix

Measured with `uv run python scripts/measure_prefix.py`, which reads the live
agent, so it cannot drift from what ships.

| Component | Chars | Ratchet | Built by |
|---|---|---|---|
| `static_instruction` | 8,064 | 8,300 (8,900 with a code executor) | `conversation/system_prompt.py` → `build_static_instruction()` |
| tool schemas (14 decls + dynamic suffix) | 11,619 | 13,600 | each tool's docstring |
| `<available_skills>` index | 1,395 | 1,400 | `tools/skill_toolset.py` |
| skills preamble | 115 | 200 | `tools/skill_toolset.py` |
| **total** | **21,193 (~4,947 real tokens)** | **23,500** | |

Every row is asserted in `tests/unit/test_prompt_budget.py`, so a regression
fails a test instead of quietly costing tokens on every call. Individual prompt
blocks are capped too: ACTING 1,200, SAFETY 1,100, STYLE 900, MEMORY 1,400.

## The prefix cache

The prefix is constant, so Vertex should serve it from cache after the first
call. Verify with `uv run python scripts/probe_context_cache.py`, which reports
the `cached_content_token_count` the model returns:

```
turn 0 ('hi')            prompt=  4947  cached=     0  (  0.0%)
turn 1 ('what is 2+2?')  prompt=  5101  cached=  4888  ( 95.8%)
turn 2 ('and 3+3?')      prompt=  5142  cached=  4888  ( 95.1%)
```

Turn 0 populates; every later turn reuses one `cache_name` with
`invocations_used` incrementing. Config is `ContextCacheConfig(min_tokens=4096,
ttl_seconds=1800, cache_intervals=10)` in `agent.py`. `min_tokens` is 4096
because `GeminiContextCacheManager`'s own per-model floor is hardcoded to 4096
for `gemini-3*`; a smaller value here is dead config.

Never put per-turn content in `system_instruction`.
`_generate_cache_fingerprint` hashes the entire string, so one varying line
invalidates the whole prefix on every turn. ADK's stock `PreloadMemoryTool`
writes `<PAST_CONVERSATIONS>` there, which kept the cache from ever forming
until `memory/preload.py`'s `HorizonPreloadMemoryTool` moved that block to the
cache-excluded contents tail. Per-turn steering (iteration count, last error,
date, todos) rides the same tail, as trailing `<system-reminder>` content from
`conversation/reminders.py`.

## What bounds growth during a session

Four mechanisms, each covering a case the others miss.

| Mechanism | Where | Bounds |
|---|---|---|
| Source caps | `tools/_output_overflow.py` (the shared limit), `tools/file_ops.py` (`_MAX_READ_CHARS`) | One call: `read` and shell output share `TERMINAL_OUTPUT_LIMIT` (51,200 chars), spilling the remainder to `lha/tool-output/` and returning a pointer. |
| Retroactive pruning | `context/tool_output_pruning.py` | Stale bulk: zeroes old large tool-result bodies before the model reads them. Floor is 500 tokens per part. Never prunes `subagent`, `clarify`, `skill`, or any `*_overflow_path`, so expensive or recoverable results survive. Off with `LHA_PRUNE_TOOL_OUTPUTS=0`. |
| Compaction | `context/summarizer.py` → `HorizonSummarizer` | Total history: ADK's `EventsCompactionConfig` triggers it; Horizon adds a REFERENCE-ONLY banner, caps each inlined tool result and call args at 2,000 chars, and tracks files cumulatively so they survive repeated summarization. |
| Preload caps | `memory/preload.py` | Recall: `LHA_PRELOAD_MAX_MEMORIES` and `LHA_PRELOAD_MAX_CHARS`, because ADK's `search_memory` has no `top_k`. |

Source caps and the pruner cover different failures. A cap bounds the worst
single call; the pruner reclaims a long tail of mid-sized results that were each
individually fine. Dropping either one leaves a real session unbounded.

## Keeping the schema surface small

Tool declarations are the largest single component, so two settings matter:

- `ADK_DISABLE_JSON_SCHEMA_FOR_FUNC_DECL=1` (set in `agent.py`). ADK's pydantic
  path emits `title` on every parameter, `default: null`, and `anyOf[X, null]`.
  The legacy path expresses the same parameters without them, worth 2,812 chars.
- `context/schema_normalization.py` runs `inspect.cleandoc` over tool
  descriptions, since the legacy path otherwise ships raw indented docstrings.

Docstring budgets are enforced per tool (400 chars simple, 900 dispatch) by
`test_prompt_budget.py`. A new tool costs roughly 500 to 900 chars of prefix on
every call, for as long as it exists.

## Checklist for a change that touches the prompt or a tool

1. `uv run pytest tests/unit/test_prompt_budget.py` for the ratchets.
2. `uv run python scripts/measure_prefix.py` to see the new composition.
3. If you touched `system_instruction` assembly, re-run
   `scripts/probe_context_cache.py` and confirm turn 1 is still around 95%.
4. Adding a tool means adding it to `tools/names.py`; the registry tests fail
   otherwise.
