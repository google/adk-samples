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

"""Convert tests/eval/evalsets/*.evalset.json (ADK conversation format) into
tests/eval/datasets/*.json (agents-cli's EvalCase format), so
`agents-cli eval run --dataset tests/eval/datasets/<name>.json` works.

agents-cli 1.3.x's `eval generate` requires each case to carry either a
top-level `prompt` (single user message) or `agent_data.turns` ending in a
user event (continued conversation) -- see
google.agents.cli.eval.cmd_generate.split_case_history. horizon's evalsets
predate that shape and use ADK's own `conversation[].user_content` format,
so `agents-cli eval run` rejects every evalset in this repo outright.

Single-invocation cases map straight to `prompt`. Multi-invocation cases map
to `agent_data.turns`, one turn per non-final invocation holding only that
invocation's user message.

Known gap: no evalset here records a `final_response` for a non-final
invocation (ADK's format never captured one), so a prior turn's assistant
reply cannot be reconstructed. Fabricating placeholder assistant text would
plant a false transcript in front of the grader, so this converter leaves
each non-final turn as a user-authored event only (no model event) --
`split_case_history` only requires the trailing event to be user-authored,
not strict user/model alternation before it. This preserves every fact the
user stated across turns (sufficient for same-session recall evals that
read the raw transcript, e.g. memory_recall's timezone case) but does not
replay any tool call an earlier turn's real response would have triggered
(e.g. `add_memory` actually persisting to Memory Bank). If a future evalset
does carry an invocation-level `final_response` string, this converter
uses it as that turn's real assistant reply.

Second known gap: 7 cases across guardrail_halt / slash_commands_and_reload /
workspace_window / safety pre-seed `session_input.state` (e.g. `halt_reason`,
`_policy_grants`) that horizon's callbacks read at turn start.
`vertexai._genai.types.common.EvalCase` has no field for arbitrary initial
session state, so this converter cannot represent it. These cases are still
emitted (never dropped silently) but PRINT_STATE_WARNING lists them, and the
case will not exercise its intended pre-seeded-state setup once converted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALSETS_DIR = REPO_ROOT / "tests" / "eval" / "evalsets"
DATASETS_DIR = REPO_ROOT / "tests" / "eval" / "datasets"


def _user_content(invocation: dict) -> dict:
    content = invocation.get("user_content") or {}
    return {"role": "user", "parts": content.get("parts", [])}


def _rubric_group(invocations: list[dict]) -> dict | None:
    rubrics = []
    for invocation in invocations:
        for rubric in invocation.get("rubrics") or []:
            rc = rubric.get("rubric_content") or {}
            rubrics.append(
                {
                    "rubric_id": rubric.get("rubric_id"),
                    "content": {
                        "property": {"text_property": rc.get("text_property")}
                    },
                }
            )
    if not rubrics:
        return None
    # Best-effort only: the eval_config.json metric in this repo
    # (rubric_based_final_response_quality_v1) is metric-level and does not
    # read rubric_groups off the case, confirmed against cmd_grade.py (no
    # "rubric" reference at all). Only google.agents.cli.eval.optimize_utils
    # (the `eval optimize` / GEPA path) reads case-level rubric_groups.
    return {"case_rubrics": {"rubrics": rubrics}}


def convert_case(case: dict) -> tuple[dict, list[str]]:
    """Convert one ADK-format eval_case into agents-cli's EvalCase shape.

    Returns (converted_case, state_warnings) -- state_warnings is non-empty
    when session_input.state could not be carried over.
    """
    invocations = case.get("conversation") or []
    if not invocations:
        raise ValueError(
            f"eval_case {case.get('eval_id')!r} has no conversation"
        )

    converted: dict = {"eval_case_id": case.get("eval_id")}

    if len(invocations) == 1:
        converted["prompt"] = _user_content(invocations[0])
    else:
        turns = []
        for i, invocation in enumerate(invocations):
            events = [{"author": "user", "content": _user_content(invocation)}]
            # Forward-compat: use a real recorded reply if one ever exists.
            final_response = invocation.get("final_response")
            if i < len(invocations) - 1 and final_response:
                events.append(
                    {
                        "author": "model",
                        "content": {
                            "role": "model",
                            "parts": [{"text": final_response}],
                        },
                    }
                )
            turns.append(
                {"turn_index": i, "turn_id": f"turn_{i}", "events": events}
            )
        converted["agent_data"] = {"turns": turns}

    rubric_group = _rubric_group(invocations)
    if rubric_group:
        converted["rubric_groups"] = rubric_group

    warnings = []
    state = (case.get("session_input") or {}).get("state") or {}
    if state:
        warnings.append(
            f"{case.get('eval_id')}: session_input.state {sorted(state)} has "
            "no home in agents-cli's EvalCase schema and was dropped; this "
            "case will run without its pre-seeded session state"
        )

    return converted, warnings


def convert_evalset(evalset_path: Path) -> tuple[dict, list[str]]:
    data = json.loads(evalset_path.read_text())
    cases = []
    all_warnings: list[str] = []
    for case in data.get("eval_cases", []):
        converted, warnings = convert_case(case)
        cases.append(converted)
        all_warnings.extend(warnings)
    return {"eval_cases": cases}, all_warnings


def _default_evalsets() -> list[Path]:
    return sorted(EVALSETS_DIR.glob("*.evalset.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evalsets",
        nargs="*",
        type=Path,
        help="Evalset JSON files to convert (default: all tests/eval/evalsets/*.evalset.json)",
    )
    args = parser.parse_args(argv)

    evalsets = args.evalsets or _default_evalsets()
    if not evalsets:
        print(f"No evalsets found under {EVALSETS_DIR}", file=sys.stderr)
        return 1

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    all_warnings: list[str] = []
    for evalset_path in evalsets:
        dataset, warnings = convert_evalset(evalset_path)
        out_path = DATASETS_DIR / (
            evalset_path.stem.replace(".evalset", "") + ".json"
        )
        out_path.write_text(json.dumps(dataset, indent=2) + "\n")
        print(
            f"{evalset_path.name} -> {out_path} ({len(dataset['eval_cases'])} cases)"
        )
        all_warnings.extend(warnings)

    if all_warnings:
        print(
            "\nWarnings (session_input.state not representable):",
            file=sys.stderr,
        )
        for warning in all_warnings:
            print(f"  - {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
