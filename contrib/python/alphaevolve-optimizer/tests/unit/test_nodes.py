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
"""Unit tests for the verifier, selector and workflow nodes. No model calls."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.adk import Workflow
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import FunctionNode
from google.genai import types

from app import population as pop
from app import problem
from app.agent import (
    CONTINUE,
    DONE,
    Candidate,
    build_report,
    create_workflow,
    deliver,
    init,
    render_report,
    report,
    select,
    verify,
    verify_program,
)
from app.sandbox import MAX_CODE_CHARS, run_candidate

# Picks, per cluster, the single node type whose first-fit-decreasing packing
# is cheapest. A realistic "right-size the node pool" improvement.
RIGHT_SIZED_POOL = """\
def place(pods, node_types):
    best = None
    for t in node_types:
        order = sorted(range(len(pods)), key=lambda i: -max(
            pods[i]["cpu"] / t["cpu"], pods[i]["memory"] / t["memory"]))
        nodes, cpu, mem = [], [], []
        for i in order:
            for n in range(len(nodes)):
                if (cpu[n] + pods[i]["cpu"] <= t["cpu"]
                        and mem[n] + pods[i]["memory"] <= t["memory"]
                        and len(nodes[n]["pods"]) < t["max_pods"]):
                    nodes[n]["pods"].append(i)
                    cpu[n] += pods[i]["cpu"]
                    mem[n] += pods[i]["memory"]
                    break
            else:
                nodes.append({"node_type": t["name"], "pods": [i]})
                cpu.append(pods[i]["cpu"])
                mem.append(pods[i]["memory"])
        cost = len(nodes) * t["hourly_cost"]
        if best is None or cost < best[0]:
            best = (cost, nodes)
    return best[1]
"""

ONE_NODE_PER_POD = (
    "def place(pods, node_types):\n"
    "    return [{'node_type': 'std-16', 'pods': [i]} for i in range(len(pods))]\n"
)


INVALID = "def place(pods, node_types):\n    return []\n"


@pytest.fixture
def ctx() -> SimpleNamespace:
    """Stands in for google.adk.Context: function nodes only use ctx.state."""
    return SimpleNamespace(state={})


@pytest.fixture(autouse=True)
def _small_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_EVALUATIONS", "2")
    monkeypatch.setenv("POPULATION_SIZE", "3")


# --- problem / data ---------------------------------------------------------


def test_bundled_workloads_load_one_instance_per_cluster() -> None:
    names = [i["name"] for i in problem.INSTANCES]
    assert names == ["ecommerce-prod", "data-platform", "ml-serving"]
    assert all(i["args"]["pods"] for i in problem.INSTANCES)


def test_custom_workloads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "w.csv"
    csv_path.write_text(
        "cluster,workload,replicas,cpu_millicores,memory_mib\n"
        "c1,api,3,500,1024\n"
        "c2,db,1,2000,8192\n"
    )
    instances = problem.load_instances(csv_path)
    assert [len(i["args"]["pods"]) for i in instances] == [3, 1]


# --- verifier ---------------------------------------------------------------


def test_baseline_is_valid_and_above_lower_bound() -> None:
    result = verify_program(problem.BASELINE_PROGRAM)
    assert result["valid"], result
    assert result["score"] >= problem.lower_bound()
    assert result["monthly_cost"] == pytest.approx(
        result["score"] * problem.HOURS_PER_MONTH, abs=0.01
    )


def test_right_sized_pool_is_cheaper_than_baseline() -> None:
    baseline = verify_program(problem.BASELINE_PROGRAM)
    improved = verify_program(RIGHT_SIZED_POOL)
    assert improved["valid"], improved
    assert improved["score"] < baseline["score"]


def test_wasteful_but_valid_plan_scores_worse() -> None:
    result = verify_program(ONE_NODE_PER_POD)
    assert result["valid"], result
    assert result["score"] > verify_program(problem.BASELINE_PROGRAM)["score"]


def test_overcommitted_node_is_rejected() -> None:
    code = (
        "def place(pods, node_types):\n"
        "    return [{'node_type': 'std-4', 'pods': list(range(len(pods)))}]\n"
    )
    result = verify_program(code)
    assert not result["valid"]
    assert "cpu" in result["error"] or "pods >" in result["error"]


def test_unknown_node_type_is_rejected() -> None:
    code = (
        "def place(pods, node_types):\n"
        "    return [{'node_type': 'gpu-xl', 'pods': [i]} for i in range(len(pods))]\n"
    )
    result = verify_program(code)
    assert not result["valid"]
    assert "node_type" in result["error"]


def test_missing_pods_are_rejected() -> None:
    result = verify_program("def place(pods, node_types):\n    return []\n")
    assert not result["valid"]
    assert "not placed" in result["error"]


def test_duplicate_pods_are_rejected() -> None:
    code = ONE_NODE_PER_POD.replace(
        "for i in range(len(pods))]", "for i in range(len(pods))] * 2"
    )
    result = verify_program(code)
    assert not result["valid"]
    assert "more than once" in result["error"]


def test_crashing_candidate_reports_error() -> None:
    result = verify_program(
        "def place(pods, node_types):\n    raise ValueError('boom')\n"
    )
    assert not result["valid"]
    assert "boom" in result["error"]


def test_infinite_loop_times_out() -> None:
    run = run_candidate(
        "def place(pods, node_types):\n    while True:\n        pass\n",
        problem.ENTRY_POINT,
        problem.INSTANCES,
        timeout_seconds=2,
    )
    assert not run["ok"]


def test_candidate_prints_do_not_corrupt_output() -> None:
    assert verify_program("print('noise')\n" + RIGHT_SIZED_POOL)["valid"]


def test_oversized_code_is_rejected() -> None:
    run = run_candidate("#" * 30_000, problem.ENTRY_POINT, problem.INSTANCES)
    assert not run["ok"]
    assert "exceeds" in run["error"]


# --- selector ---------------------------------------------------------------


def test_population_keeps_best_valid_candidates() -> None:
    population: list[dict] = []
    for cid, score in enumerate([5.0, 4.0, 6.0, 4.5]):
        population = pop.add_candidate(
            population, {"id": cid, "valid": True, "score": score}, max_size=2
        )
    assert [c["score"] for c in population] == [4.0, 4.5]


def test_population_ignores_invalid_candidates() -> None:
    population = pop.add_candidate(
        [], {"id": 0, "valid": False, "score": None}, max_size=2
    )
    assert population == []
    assert pop.best(population) is None


# --- function nodes ---------------------------------------------------------


def _verify_and_select(ctx: SimpleNamespace, idea: str, code: str):
    checked = verify(ctx, Candidate(idea=idea, code=code))
    return checked, select(ctx, checked.output)


def test_init_seeds_baseline_and_budget(ctx: SimpleNamespace) -> None:
    event = init(ctx, "prefer fewer node types")
    brief = event.output
    baseline = verify_program(problem.BASELINE_PROGRAM)
    assert brief["baseline_monthly_cost"] == baseline["monthly_cost"]
    assert brief["best_monthly_cost"] == baseline["monthly_cost"]
    assert brief["evaluations_remaining"] == 2
    assert brief["operator_request"] == "prefer fewer node types"
    assert len(brief["clusters"]) == len(problem.INSTANCES)
    assert [p["id"] for p in brief["parents"]] == [0]
    assert brief["last_feedback"] is None
    assert brief["max_code_chars"] == MAX_CODE_CHARS
    assert brief["parents"][0]["code_chars"] == len(problem.BASELINE_PROGRAM)
    assert "Baseline" in event.content.parts[0].text


def test_init_resets_a_previous_run(ctx: SimpleNamespace) -> None:
    init(ctx)
    _verify_and_select(ctx, "right-size pool", RIGHT_SIZED_POOL)
    init(ctx)
    assert ctx.state["evolve_evaluations"] == 0
    assert len(ctx.state["evolve_population"]) == 1
    assert len(ctx.state["evolve_history"]) == 1


def test_better_candidate_becomes_best_and_loop_continues(
    ctx: SimpleNamespace,
) -> None:
    init(ctx)
    checked, routed = _verify_and_select(ctx, "right-size", RIGHT_SIZED_POOL)
    assert checked.output["valid"]
    assert routed.actions.route == CONTINUE
    brief = routed.output
    assert brief["best_monthly_cost"] == checked.output["monthly_cost"]
    assert brief["best_monthly_cost"] < brief["baseline_monthly_cost"]
    assert brief["parents"][0]["code"] == RIGHT_SIZED_POOL
    assert brief["last_feedback"]["id"] == 1
    assert brief["evaluations_remaining"] == 1


def test_invalid_candidate_feeds_back_error(ctx: SimpleNamespace) -> None:
    init(ctx)
    checked, routed = _verify_and_select(ctx, "bad", INVALID)
    assert not checked.output["valid"]
    assert "not placed" in routed.output["last_feedback"]["error"]
    assert len(ctx.state["evolve_population"]) == 1


def test_duplicate_code_is_rejected_without_running(
    ctx: SimpleNamespace,
) -> None:
    init(ctx)
    checked = verify(ctx, Candidate(idea="same", code=problem.BASELINE_PROGRAM))
    assert not checked.output["valid"]
    assert "identical to candidate 0" in checked.output["error"]


def test_budget_exhaustion_routes_to_report(ctx: SimpleNamespace) -> None:
    init(ctx)
    _, first = _verify_and_select(ctx, "bad", INVALID)
    _, second = _verify_and_select(ctx, "right-size", RIGHT_SIZED_POOL)
    assert first.actions.route == CONTINUE
    assert second.actions.route == DONE
    assert second.output["best_id"] == 2


def test_reaching_lower_bound_stops_early(
    ctx: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    init(ctx)
    monkeypatch.setattr(problem, "lower_bound", lambda: 1e9)
    _, routed = _verify_and_select(ctx, "right-size", RIGHT_SIZED_POOL)
    assert routed.actions.route == DONE
    assert routed.output["reached_bound"]


def test_report_uses_verified_numbers(ctx: SimpleNamespace) -> None:
    init(ctx)
    _verify_and_select(ctx, "right-size pool", RIGHT_SIZED_POOL)
    numbers = build_report(ctx.state)
    assert numbers["monthly_savings"] > 0
    assert (
        numbers["best_monthly_cost"]
        == verify_program(RIGHT_SIZED_POOL)["monthly_cost"]
    )
    assert {c["cluster"] for c in numbers["clusters"]} == {
        i["name"] for i in problem.INSTANCES
    }
    text = render_report(numbers)
    assert f"${numbers['monthly_savings']:,.2f}" in text
    assert "```python" in text
    event = report(ctx)
    assert "best_code" not in event.output
    assert event.output["monthly_savings"] == (
        f"${numbers['monthly_savings']:,.2f}"
    )
    final = deliver(ctx, "### Summary\nShort.")
    assert final.output.startswith(text)
    assert final.output.endswith("### Summary\nShort.")


# --- workflow graph ---------------------------------------------------------


def _stub_workflow(codes: list[str], seen: list[dict]) -> Workflow:
    def propose(node_input: dict) -> dict:
        seen.append(node_input)
        return {"idea": f"idea {len(seen)}", "code": codes[len(seen) - 1]}

    def narrate(node_input: dict) -> str:
        return f"saved {node_input['monthly_savings']}"

    return create_workflow(
        proposer=FunctionNode(func=propose, name="propose"),
        narrator=FunctionNode(func=narrate, name="narrate"),
    )


async def _run(workflow: Workflow) -> tuple[list, dict]:
    runner = InMemoryRunner(app=App(root_agent=workflow, name="app"))
    session = await runner.session_service.create_session(
        app_name="app", user_id="u"
    )
    message = types.Content(role="user", parts=[types.Part(text="cut cost")])
    events = [
        e
        async for e in runner.run_async(
            user_id="u", session_id=session.id, new_message=message
        )
    ]
    session = await runner.session_service.get_session(
        app_name="app", user_id="u", session_id=session.id
    )
    return events, session.state


def test_workflow_loops_until_budget_then_reports() -> None:
    seen: list[dict] = []
    events, state = asyncio.run(
        _run(_stub_workflow([INVALID, RIGHT_SIZED_POOL], seen))
    )
    assert len(seen) == 2
    assert seen[1]["last_feedback"]["valid"] is False
    routes = [e.actions.route for e in events if e.actions.route]
    assert routes == [CONTINUE, DONE]
    assert state["evolve_evaluations"] == 2
    final = [e for e in events if e.output is not None][-1]
    assert final.output.startswith("## Kubernetes node cost plan")
    assert final.output.endswith(f"saved {_savings(state)}")
    assert final.content.parts[0].text == final.output


def _savings(state: dict) -> str:
    return f"${build_report(state)['monthly_savings']:,.2f}"
