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
"""Unit tests for the verifier, selector and agent tools. No model calls."""

from types import SimpleNamespace

import pytest

from app import population as pop
from app import problem
from app.agent import (
    evaluate_candidate,
    get_parents,
    get_problem,
    verify_program,
)
from app.sandbox import run_candidate

FIRST_FIT_DECREASING = """\
def pack(items, capacity):
    order = sorted(range(len(items)), key=lambda i: -items[i])
    bins, loads = [], []
    for i in order:
        for b, load in enumerate(loads):
            if load + items[i] <= capacity:
                bins[b].append(i)
                loads[b] += items[i]
                break
        else:
            bins.append([i])
            loads.append(items[i])
    return bins
"""


@pytest.fixture
def tool_context() -> SimpleNamespace:
    return SimpleNamespace(state={})


@pytest.fixture(autouse=True)
def _small_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_EVALUATIONS", "2")
    monkeypatch.setenv("POPULATION_SIZE", "3")


# --- verifier ---------------------------------------------------------------


def test_baseline_is_valid_and_above_lower_bound() -> None:
    result = verify_program(problem.BASELINE_PROGRAM)
    assert result["valid"], result
    assert result["score"] >= problem.lower_bound()


def test_better_heuristic_scores_lower_than_baseline() -> None:
    baseline = verify_program(problem.BASELINE_PROGRAM)
    improved = verify_program(FIRST_FIT_DECREASING)
    assert improved["valid"], improved
    assert improved["score"] <= baseline["score"]


def test_overfull_bin_is_rejected() -> None:
    result = verify_program(
        "def pack(items, capacity):\n    return [list(range(len(items)))]\n"
    )
    assert not result["valid"]
    assert "exceeds capacity" in result["error"]


def test_missing_items_are_rejected() -> None:
    result = verify_program("def pack(items, capacity):\n    return []\n")
    assert not result["valid"]
    assert "not packed" in result["error"]


def test_duplicate_items_are_rejected() -> None:
    code = "def pack(items, capacity):\n    return [[i] for i in range(len(items))] + [[0]]\n"
    result = verify_program(code)
    assert not result["valid"]
    assert "more than once" in result["error"]


def test_crashing_candidate_reports_error() -> None:
    result = verify_program(
        "def pack(items, capacity):\n    raise ValueError('boom')\n"
    )
    assert not result["valid"]
    assert "boom" in result["error"]


def test_syntax_error_reports_error() -> None:
    result = verify_program("def pack(items, capacity) return []\n")
    assert not result["valid"]
    assert result["error"]


def test_infinite_loop_times_out() -> None:
    run = run_candidate(
        "def pack(items, capacity):\n    while True:\n        pass\n",
        problem.ENTRY_POINT,
        problem.INSTANCES,
        timeout_seconds=2,
    )
    assert not run["ok"]


def test_candidate_prints_do_not_corrupt_output() -> None:
    code = "print('noise')\n" + FIRST_FIT_DECREASING
    assert verify_program(code)["valid"]


def test_oversized_code_is_rejected() -> None:
    run = run_candidate("#" * 30_000, problem.ENTRY_POINT, problem.INSTANCES)
    assert not run["ok"]
    assert "exceeds" in run["error"]


# --- selector ---------------------------------------------------------------


def test_population_keeps_best_valid_candidates() -> None:
    population: list[dict] = []
    for cid, score in enumerate([50, 40, 60, 45]):
        population = pop.add_candidate(
            population, {"id": cid, "valid": True, "score": score}, max_size=2
        )
    assert [c["score"] for c in population] == [40, 45]


def test_population_ignores_invalid_candidates() -> None:
    population = pop.add_candidate(
        [], {"id": 0, "valid": False, "score": None}, max_size=2
    )
    assert population == []
    assert pop.best(population) is None


# --- agent tools ------------------------------------------------------------


def test_get_problem_seeds_baseline(tool_context: SimpleNamespace) -> None:
    info = get_problem(tool_context)
    assert (
        info["best_score"] == verify_program(problem.BASELINE_PROGRAM)["score"]
    )
    assert info["lower_bound"] == problem.lower_bound()
    assert len(get_parents(tool_context)["parents"]) == 1


def test_evaluate_candidate_updates_population_and_budget(
    tool_context: SimpleNamespace,
) -> None:
    get_problem(tool_context)
    first = evaluate_candidate(
        FIRST_FIT_DECREASING, "sort decreasing", tool_context
    )
    assert first["valid"]
    assert first["evaluations_remaining"] == 1
    assert (
        get_parents(tool_context)["parents"][0]["score"] == first["best_score"]
    )

    invalid = evaluate_candidate(
        "def pack(items, capacity):\n    return []\n", "bad", tool_context
    )
    assert not invalid["valid"]
    assert invalid["done"]

    exhausted = evaluate_candidate(FIRST_FIT_DECREASING, "again", tool_context)
    assert exhausted["budget_exhausted"]
