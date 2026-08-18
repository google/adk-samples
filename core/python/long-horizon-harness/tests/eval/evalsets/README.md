# Evaluation Sets

This directory contains evaluation sets for testing agent behavior, in ADK's own
evaluation format. `agents-cli eval run` reads a different `EvalCase` shape
(`prompt` or `agent_data.turns`, see `google.agents.cli.eval.cmd_generate`), so
evalsets here are converted to that shape before running.

## Running Evaluations

```bash
# Convert every evalset in this directory into tests/eval/datasets/*.json
uv run python scripts/evalset_to_dataset.py

# Run one converted dataset
agents-cli eval run --dataset tests/eval/datasets/smoke.json --config ../eval_config.json

# Convert and run a single evalset
uv run python scripts/evalset_to_dataset.py tests/eval/evalsets/custom.evalset.json
agents-cli eval run --dataset tests/eval/datasets/custom.json --config ../eval_config.json
```

See `scripts/evalset_to_dataset.py`'s module docstring for the two known
conversion gaps: multi-turn cases carry no synthesized assistant reply for
earlier turns (none is recorded in this format), and a handful of cases that
pre-seed `session_input.state` (guardrail_halt, slash_commands_and_reload,
workspace_window, safety) lose that seed, since `EvalCase` has no field for
arbitrary initial session state. The converter prints a warning per affected
case rather than dropping it silently.

## Evalset Format

Each `.evalset.json` follows the ADK evaluation format:

```json
{
  "eval_set_id": "unique_id",
  "name": "Human-readable name",
  "description": "What this evalset tests",
  "eval_cases": [
    {
      "eval_id": "case_id",
      "conversation": [
        {
          "user_content": {
            "parts": [{"text": "User message"}]
          },
          "intermediate_data": {
            "tool_uses": [
              {"name": "tool_name", "args": {"param": "value"}}
            ]
          }
        }
      ],
      "session_input": {
        "app_name": "app_name",
        "user_id": "test_user",
        "state": {}
      }
    }
  ]
}
```

## Key Fields

- `eval_cases`: Array of test scenarios
- `conversation`: Sequence of user messages
- `intermediate_data.tool_uses`: Expected tool calls (for trajectory matching)
- `session_input`: Initial session state

## Evaluation Metrics

This repo's `../eval_config.json` declares one metric,
`rubric_based_final_response_quality_v1` (an LLM judge scored against
per-metric rubrics, threshold 0.8). There is no trajectory grader --
`intermediate_data.tool_uses` below is declarative documentation of the
expected trajectory, not something the grader checks.

## Creating Custom Evalsets

1. Copy `basic.evalset.json` as a template
2. Add cases based on your agent's scenarios
3. Include expected tool calls as documentation (see Evaluation Metrics above)
4. Run `uv run python scripts/evalset_to_dataset.py tests/eval/evalsets/your_evalset.json`
5. Run `agents-cli eval run --dataset tests/eval/datasets/your_evalset.json --config ../eval_config.json`

## Tips

- Start with 3-5 representative cases
- Include both happy path and edge cases
- Test each core capability of your agent
- Add cases when you find bugs in production

See [ADK documentation](https://google.github.io/adk-docs/) for advanced evaluation options.
