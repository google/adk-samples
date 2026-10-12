#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a live-eval-harness ``record_to`` JSON dump into the trace format
``agents-cli eval grade`` consumes, then invoke it (project-free mode).

This is an INFORMATIONAL reporting layer on top of the stage-4/stage-6 live
eval harness (``tests/integration/eval_harness.py``) and its recorder
(``tests/integration/record_actuals.py``) — it does NOT gate anything. It
lets a human compare our own harness's pass/fail verdicts against
agents-cli's LLM-judged `tool_use_quality` / `final_response_quality`
metrics on the SAME recorded run.

Usage:
    python scripts/eval_grade_report.py <record_actuals.json> \\
        [--traces-out artifacts/traces/traces_report.json] \\
        [--output artifacts/grade_results] \\
        [--config artifacts/grade_results/eval_config.yaml] \\
        [--metrics tool_use_quality,final_response_quality]

Requires the ``agents-cli`` binary on PATH (installed as a uv tool, v1.1.0 at
the time this script was written) and a real GCP project reachable for the
Vertex AI Eval Service (built-in metrics are graded server-side — see
research/agents-cli-architecture.md §4). Env resolution is the caller's
responsibility (the ``test-live-report`` Makefile target uses the same
``app/.env`` bootstrap as ``make test-live``).

Trace format (verified against installed google-agents-cli 1.1.0 /
google-cloud-aiplatform 1.162.0 on 2026-07-23 — see
``research/agents-cli-architecture.md`` §2 and the Task 15 addendum):
``vertexai._genai.types.common.EvaluationDataset`` containing a list of
``EvalCase`` — each case's ``prompt`` (user Content), ``responses[0].response``
(final-answer Content), and ``agent_data.turns[0].events`` (one
``AgentEvent`` per recorded tool call, each carrying a ``function_call``
Part) is exactly what ``tool_use_quality`` and ``final_response_quality``
read (``_build_evaluation_instance`` / ``_eval_case_to_agent_data`` in
``vertexai/_genai/_evals_metric_handlers.py``).
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path so ``from app.agent import root_agent`` works when this
# script is invoked directly (``python scripts/eval_grade_report.py``), same
# convention as scripts/demo_assets.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The actual SDK models (not hand-rolled dicts) guarantee schema fidelity —
# this repo's own .venv already has google-cloud-aiplatform installed
# (verified 2026-07-23: v1.162.0), same package agents-cli's `eval grade`
# uses to parse the traces file back out.
from google.genai import types as genai_types
from vertexai._genai.types import evals as evals_types
from vertexai._genai.types.common import (
    EvalCase,
    EvaluationDataset,
    ResponseCandidate,
)

# Mirrors the harness's 3 dimensions with the closest available agents-cli
# built-ins (single-turn — our recorded cases are one user turn each):
# "tools"/"trajectory" -> tool_use_quality, "answer" -> final_response_quality.
# Exact names verified live against `agents-cli eval metric list` 2026-07-23
# (case-insensitive; the CLI upper()s whatever string it's given).
DEFAULT_METRICS = ["tool_use_quality", "final_response_quality"]

_DEFAULT_CONFIG_YAML = """\
# Auto-written by scripts/eval_grade_report.py (Task 15, ws16 stage 7).
# Minimal built-in-only config; INFORMATIONAL reporting layer, not a gate.
metrics_to_run:
  - tool_use_quality
  - final_response_quality
"""


def build_agent_config_map() -> (
    tuple[dict[str, evals_types.AgentConfig], str] | None
):
    """Introspect this repo's live agent graph for tool declarations.

    [Task 15 addendum finding, 2026-07-23] Without a populated
    ``agent_data.agents`` map, agents-cli's server-side adaptive-rubric
    generator assumes the agent under test has NO tools declared and scores
    any observed tool call as a violation — see the addendum in
    research/agents-cli-architecture.md for the concrete before/after
    evidence. ``AgentConfig.from_agent()`` (the SDK's own helper, also used
    by agents-cli's local inference runner) wraps each plain-callable tool in
    ADK's ``FunctionTool`` so its declaration logic strips ADK-injected
    params like ``tool_context`` — the same trick agents-cli's own
    ``_inference_runner.py`` relies on.

    Returns ``(agent_id -> AgentConfig, root_agent_id)``, or ``None`` if
    ``app.agent`` cannot be imported (best-effort enrichment; the trace is
    still gradeable without it, just with the documented fidelity gap).
    """
    try:
        from app.agent import root_agent
    except Exception as e:  # pragma: no cover - best-effort enrichment
        print(
            f"[eval_grade_report] WARNING: could not import app.agent for tool "
            f"declarations ({e}); agent_data.agents will be omitted.",
            file=sys.stderr,
        )
        return None

    graph = [root_agent, *(getattr(root_agent, "sub_agents", None) or [])]
    agents_map = {
        agent.name: evals_types.AgentConfig.from_agent(agent) for agent in graph
    }
    return agents_map, root_agent.name


def _tool_call_events(
    tool_calls: list[dict[str, Any]], root_agent_id: str
) -> list[evals_types.AgentEvent]:
    """Build one AgentEvent per recorded tool call.

    ``record_actuals.py`` doesn't tag calls with which (sub-)agent made
    them, but ``transfer_to_agent`` calls are themselves recorded and name
    their target — used here to track the acting agent across the
    trajectory so event authorship matches the real agent graph rather than
    a single flat placeholder.
    """
    events = []
    current_author = root_agent_id
    for call in tool_calls:
        name = call.get("name", "")
        part = genai_types.Part(
            function_call=genai_types.FunctionCall(
                name=name, args=call.get("args") or {}
            )
        )
        events.append(
            evals_types.AgentEvent(
                author=current_author,
                content=genai_types.Content(role="model", parts=[part]),
            )
        )
        if name == "transfer_to_agent":
            target = (call.get("args") or {}).get("agent_name")
            if target:
                current_author = target
    return events


def _case_to_eval_case(
    case: dict[str, Any],
    agents_map: dict[str, evals_types.AgentConfig] | None,
    root_agent_id: str,
) -> EvalCase | None:
    """One recorder row -> one EvalCase, or None if inference didn't succeed.

    Cases where ``inference_ok`` is False carry no tool calls / final answer
    to grade (the harness itself already fails/xfails these); skip them here
    rather than sending agents-cli an empty-response case it would reject.
    """
    if not case.get("inference_ok"):
        return None

    query = case.get("query") or ""
    final_text = case.get("final_response_text") or ""
    tool_calls = case.get("actual_tool_calls") or []

    prompt = genai_types.Content(
        role="user", parts=[genai_types.Part(text=query)]
    )
    response = genai_types.Content(
        role="model", parts=[genai_types.Part(text=final_text)]
    )
    events = _tool_call_events(tool_calls, root_agent_id)
    agent_data = (
        evals_types.AgentData(
            agents=agents_map,
            turns=[
                evals_types.ConversationTurn(
                    turn_index=0, turn_id="turn_0", events=events
                )
            ],
        )
        if events
        else None
    )

    return EvalCase(
        eval_case_id=case.get("eval_id"),
        prompt=prompt,
        responses=[ResponseCandidate(response=response)],
        agent_data=agent_data,
    )


def convert_record_to_traces(
    record_path: Path, *, include_agent_declarations: bool = True
) -> tuple[EvaluationDataset, int, int]:
    """Load a ``record_to`` JSON dump, return (dataset, n_included, n_skipped)."""
    rows = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(
            f"{record_path}: expected a JSON list (record_actuals.py output), "
            f"got {type(rows).__name__}"
        )

    agents_map: dict[str, evals_types.AgentConfig] | None = None
    root_agent_id = "root_agent"
    if include_agent_declarations:
        built = build_agent_config_map()
        if built is not None:
            agents_map, root_agent_id = built

    eval_cases = []
    skipped = 0
    for row in rows:
        eval_case = _case_to_eval_case(row, agents_map, root_agent_id)
        if eval_case is None:
            skipped += 1
            continue
        eval_cases.append(eval_case)
    return EvaluationDataset(eval_cases=eval_cases), len(eval_cases), skipped


def write_traces_file(dataset: EvaluationDataset, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        dataset.model_dump_json(exclude_none=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_default_config(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_DEFAULT_CONFIG_YAML, encoding="utf-8")


def run_agents_cli_grade(
    *,
    traces_path: Path,
    output_dir: Path,
    config_path: Path,
    metrics: str | None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "agents-cli",
        "eval",
        "grade",
        "--traces",
        str(traces_path),
        "--output",
        str(output_dir),
        "--config",
        str(config_path),
    ]
    if metrics:
        cmd += ["--metrics", metrics]
    print(f"[eval_grade_report] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)  # noqa: S603
    return result.returncode


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record_json",
        type=Path,
        help="record_actuals.py output JSON to convert",
    )
    parser.add_argument(
        "--traces-out",
        type=Path,
        default=None,
        help="Where to write the converted traces file "
        "(default: artifacts/traces/traces_<record-stem>_<ts>.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/grade_results"),
        help="Directory for agents-cli's results_<ts>.json/.html",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Metrics config YAML/JSON (default: auto-written minimal "
        "built-in config at <output>/eval_config.yaml)",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Override --metrics passed to agents-cli (comma-separated). "
        f"Default comes from the written config: {','.join(DEFAULT_METRICS)}",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only write the traces/config files; do not invoke agents-cli.",
    )
    parser.add_argument(
        "--no-agent-declarations",
        action="store_true",
        help="Skip importing app.agent to attach tool declarations "
        "(agent_data.agents) — see the Task 15 addendum on why this "
        "matters for tool_use_quality fidelity.",
    )
    args = parser.parse_args(argv)

    if not args.record_json.is_file():
        parser.error(f"record JSON not found: {args.record_json}")

    dataset, n_included, n_skipped = convert_record_to_traces(
        args.record_json,
        include_agent_declarations=not args.no_agent_declarations,
    )
    if n_included == 0:
        print(
            f"[eval_grade_report] ERROR: 0 gradeable cases in {args.record_json} "
            f"({n_skipped} skipped for inference_ok=False) — nothing to grade.",
            file=sys.stderr,
        )
        return 1

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    traces_out = args.traces_out or Path(
        f"artifacts/traces/traces_{args.record_json.stem}_{ts}.json"
    )
    write_traces_file(dataset, traces_out)
    print(
        f"[eval_grade_report] wrote {n_included} case(s) to {traces_out} "
        f"({n_skipped} skipped for inference_ok=False)"
    )

    config_path = args.config or (args.output / "eval_config.yaml")
    if not args.config:
        write_default_config(config_path)
        print(
            f"[eval_grade_report] wrote default metrics config to {config_path}"
        )
    elif not config_path.is_file():
        parser.error(f"--config file not found: {config_path}")

    if args.convert_only:
        return 0

    return run_agents_cli_grade(
        traces_path=traces_out,
        output_dir=args.output,
        config_path=config_path,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
