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

"""Round-trips real evalsets through evalset_to_dataset and asserts the
output satisfies agents-cli's own EvalCase rules (cmd_generate.py):
exactly one of prompt/agent_data.turns, trailing event is user-authored,
ids preserved, case count preserved. No Vertex/network calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evalset_to_dataset import convert_case, convert_evalset  # noqa: E402

EVALSETS_DIR = REPO_ROOT / "tests" / "eval" / "evalsets"


def _flatten_prior_events(case: dict) -> list[dict]:
    events = []
    for turn in (case.get("agent_data") or {}).get("turns") or []:
        events.extend(turn.get("events") or [])
    return events


def _assert_generate_contract(case: dict) -> None:
    # Mirrors cmd_generate.split_case_history's own validation.
    has_prompt = bool(case.get("prompt"))
    has_agent_data = bool(case.get("agent_data"))
    assert has_prompt or has_agent_data, case
    assert not (has_prompt and _flatten_prior_events(case)), (
        "prompt and agent_data.turns are mutually exclusive"
    )
    if has_agent_data:
        events = _flatten_prior_events(case)
        assert events, "agent_data.turns produced no events"
        assert events[-1]["author"] == "user", (
            "trailing event must be user-authored or split_case_history raises"
        )
        assert events[-1]["content"]["parts"], (
            "trailing user event has no content"
        )


def test_single_turn_evalset_maps_to_prompt():
    dataset, warnings = convert_evalset(EVALSETS_DIR / "smoke.evalset.json")
    assert warnings == []
    assert len(dataset["eval_cases"]) == 1
    case = dataset["eval_cases"][0]
    assert case["eval_case_id"] == "ping"
    assert "prompt" in case
    assert "agent_data" not in case
    _assert_generate_contract(case)


def test_multi_turn_evalset_maps_to_agent_data_turns():
    source = json.loads(
        (EVALSETS_DIR / "memory_recall.evalset.json").read_text()
    )
    n_source_cases = len(source["eval_cases"])
    dataset, _warnings = convert_evalset(
        EVALSETS_DIR / "memory_recall.evalset.json"
    )

    assert len(dataset["eval_cases"]) == n_source_cases

    multi_turn = [
        c for c in source["eval_cases"] if len(c["conversation"]) > 1
    ][0]
    converted = next(
        c
        for c in dataset["eval_cases"]
        if c["eval_case_id"] == multi_turn["eval_id"]
    )
    assert "prompt" not in converted
    assert "agent_data" in converted
    _assert_generate_contract(converted)

    # Last turn's event carries the trailing (live) user message verbatim.
    last_invocation_text = multi_turn["conversation"][-1]["user_content"][
        "parts"
    ][0]["text"]
    trailing_event = _flatten_prior_events(converted)[-1]
    assert trailing_event["content"]["parts"][0]["text"] == last_invocation_text

    # Earlier turns' user text survives even though no model reply is
    # fabricated for them (no evalset here records one).
    first_invocation_text = multi_turn["conversation"][0]["user_content"][
        "parts"
    ][0]["text"]
    all_texts = [
        part["text"]
        for event in _flatten_prior_events(converted)
        for part in event["content"]["parts"]
    ]
    assert first_invocation_text in all_texts


def test_case_count_preserved_across_every_evalset():
    for evalset_path in sorted(EVALSETS_DIR.glob("*.evalset.json")):
        source = json.loads(evalset_path.read_text())
        dataset, _warnings = convert_evalset(evalset_path)
        assert len(dataset["eval_cases"]) == len(source["eval_cases"]), (
            evalset_path.name
        )
        for case in dataset["eval_cases"]:
            _assert_generate_contract(case)


def test_eval_case_ids_preserved():
    source = json.loads(
        (EVALSETS_DIR / "tool_selection_core.evalset.json").read_text()
    )
    dataset, _warnings = convert_evalset(
        EVALSETS_DIR / "tool_selection_core.evalset.json"
    )
    source_ids = [c["eval_id"] for c in source["eval_cases"]]
    dataset_ids = [c["eval_case_id"] for c in dataset["eval_cases"]]
    assert source_ids == dataset_ids


def test_session_input_state_is_reported_not_silently_dropped():
    # guardrail_halt's cases seed session_input.state (halt_reason), which
    # has no home in agents-cli's EvalCase schema.
    _dataset, warnings = convert_evalset(
        EVALSETS_DIR / "guardrail_halt.evalset.json"
    )
    assert any("halt_reason" in w for w in warnings)


def test_case_with_no_conversation_raises():
    try:
        convert_case({"eval_id": "empty", "conversation": []})
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty conversation")
