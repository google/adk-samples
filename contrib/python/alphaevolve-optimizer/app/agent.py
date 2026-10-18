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
"""An AlphaEvolve-style evolutionary code optimizer built as an ADK 2.0 workflow.

Applied to a real platform-engineering decision: evolving a Kubernetes
workload placement program that cuts node cost while respecting every pod's
CPU and memory requests.

The loop is an explicit graph, so the model only does the one thing a model
is needed for (writing a new program) and everything else is deterministic::

    START -> init -> propose -> verify -> select --continue--> propose
                                            |
                                            +--done--> report -> narrate -> deliver

- init (function): seeds the baseline, describes the clusters and sets the
  evaluation budget in session state.
- propose (single_turn Agent): writes ONE new program from the problem, the
  best verified parents and the last verifier feedback.
- verify (function): runs the program in a resource-limited subprocess and
  scores it in this process. No model involved.
- select (function): updates the population and routes to another proposal
  or to the report, bounded by MAX_EVALUATIONS and the proven lower bound.
- report (function): re-verifies the baseline and the best program and
  builds the savings report from verifier numbers only.
- narrate (single_turn Agent): adds a short summary for a platform team,
  quoting only numbers that are in the report.
- deliver (function): sends the report and the summary as one final answer.
"""

import hashlib
import os
from typing import Any

from google.adk import Agent, Context, Event, Workflow
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.workflow import BaseNode
from google.genai import types
from pydantic import BaseModel, Field

from . import population as pop
from . import problem
from .sandbox import MAX_CODE_CHARS, run_candidate

_POPULATION_KEY = "evolve_population"
_HISTORY_KEY = "evolve_history"
_EVALUATIONS_KEY = "evolve_evaluations"
_REQUEST_KEY = "evolve_request"
_SEEN_KEY = "evolve_seen_code"
_REPORT_KEY = "evolve_report"

CONTINUE = "continue"
DONE = "done"

# Long verifier errors are trimmed before they go back to the model.
_MAX_ERROR_CHARS = 400


def _required_env(name: str, value: str | None) -> str:
    """Validates a setting declared in .env.example, failing clearly if unset."""
    if not value:
        raise RuntimeError(f"{name} is not set; copy .env.example to .env.")
    return value


def _max_evaluations() -> int:
    return int(_required_env("MAX_EVALUATIONS", os.getenv("MAX_EVALUATIONS")))


def _population_size() -> int:
    return int(_required_env("POPULATION_SIZE", os.getenv("POPULATION_SIZE")))


def _timeout_seconds() -> float:
    return float(
        _required_env(
            "CANDIDATE_TIMEOUT_SECONDS", os.getenv("CANDIDATE_TIMEOUT_SECONDS")
        )
    )


def verify_program(code: str) -> dict[str, Any]:
    """Runs and scores a program against the benchmark. No model involved."""
    run = run_candidate(
        code,
        problem.ENTRY_POINT,
        problem.INSTANCES,
        timeout_seconds=_timeout_seconds(),
    )
    if not run["ok"]:
        return {"valid": False, "score": None, "error": run["error"]}
    return problem.score_outputs(run["outputs"])


def _monthly(hourly: float) -> float:
    return round(hourly * problem.HOURS_PER_MONTH, 2)


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Schemas for the data that flows between nodes
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    """One program proposed by the model."""

    idea: str = Field(
        description=(
            "One sentence: what changed versus the parent and why it should"
            " be cheaper."
        )
    )
    code: str = Field(
        description=(
            "Complete Python source defining place(pods, node_types). Standard"
            " library only, deterministic."
        )
    )


class ProposalBrief(BaseModel):
    """Everything the proposer sees for one round. Built by init and select."""

    problem: str
    node_types: list[dict[str, Any]]
    clusters: list[dict[str, Any]]
    lower_bound_monthly_cost: float
    baseline_monthly_cost: float
    best_monthly_cost: float
    parents: list[dict[str, Any]] = Field(
        description="Best verified programs so far, cheapest first, with code."
    )
    tried: list[dict[str, Any]] = Field(
        description="Every idea tried so far with its verified result."
    )
    last_feedback: dict[str, Any] | None = Field(
        default=None,
        description="Verifier result for the latest candidate, if any.",
    )
    evaluations_remaining: int
    max_code_chars: int = Field(
        description="Programs longer than this are rejected unrun."
    )
    operator_request: str = ""


def _brief(state: Any) -> dict[str, Any]:
    """Builds the proposer's input from session state."""
    population = sorted(state.get(_POPULATION_KEY, []), key=pop.rank_key)
    history = state.get(_HISTORY_KEY, [])
    baseline = next(c for c in history if c["id"] == 0)
    best = population[0]
    last = history[-1] if len(history) > 1 else None
    return ProposalBrief(
        problem=problem.PROBLEM_STATEMENT,
        node_types=problem.NODE_TYPES,
        clusters=problem.describe_workloads(),
        lower_bound_monthly_cost=_monthly(problem.lower_bound()),
        baseline_monthly_cost=baseline["monthly_cost"],
        best_monthly_cost=_monthly(best["score"]),
        parents=[
            {
                "id": c["id"],
                "monthly_cost": _monthly(c["score"]),
                "idea": c["idea"],
                "code_chars": len(c["code"]),
                "code": c["code"],
            }
            for c in population
        ],
        tried=[
            {
                "id": h["id"],
                "idea": h["idea"],
                "valid": h["valid"],
                "monthly_cost": h["monthly_cost"],
                "code_chars": h["code_chars"],
                "error": h["error"],
            }
            for h in history
        ],
        last_feedback=last,
        evaluations_remaining=_max_evaluations()
        - state.get(_EVALUATIONS_KEY, 0),
        max_code_chars=MAX_CODE_CHARS,
        operator_request=state.get(_REQUEST_KEY, ""),
    ).model_dump()


def _history_entry(
    candidate_id: int, idea: str, code: str, result: dict[str, Any]
) -> dict[str, Any]:
    error = result.get("error")
    return {
        "id": candidate_id,
        "idea": idea,
        "code_chars": len(code),
        "valid": bool(result["valid"]),
        "monthly_cost": (
            _monthly(result["score"]) if result["valid"] else None
        ),
        "per_cluster": result.get("per_instance"),
        "error": error[:_MAX_ERROR_CHARS] if error else None,
    }


# ---------------------------------------------------------------------------
# Function nodes (deterministic, no model calls)
# ---------------------------------------------------------------------------


def init(ctx: Context, node_input: str = "") -> Event:
    """Starts a fresh run: seeds the baseline and sets the budget.

    Every user message starts a new optimization run, so state from an
    earlier run in the same session is replaced.
    """
    baseline = verify_program(problem.BASELINE_PROGRAM)
    if not baseline["valid"]:
        raise RuntimeError(f"Baseline program is invalid: {baseline['error']}")
    idea = "baseline: single std-8 node pool, first-fit"
    ctx.state[_POPULATION_KEY] = [
        {
            "id": 0,
            "idea": idea,
            "code": problem.BASELINE_PROGRAM,
            "valid": True,
            "score": baseline["score"],
        }
    ]
    ctx.state[_SEEN_KEY] = {_code_hash(problem.BASELINE_PROGRAM): 0}
    ctx.state[_HISTORY_KEY] = [
        _history_entry(0, idea, problem.BASELINE_PROGRAM, baseline)
    ]
    ctx.state[_EVALUATIONS_KEY] = 0
    ctx.state[_REQUEST_KEY] = (node_input or "").strip()[:1000]
    budget = _max_evaluations()
    clusters = problem.describe_workloads()
    pods = sum(c["pods"] for c in clusters)
    return Event(
        message=(
            f"Optimizing node cost for {len(clusters)} clusters ({pods} pods)."
            f" Baseline: ${baseline['monthly_cost']:,.2f}/month. Proven lower"
            f" bound: ${_monthly(problem.lower_bound()):,.2f}/month. Budget:"
            f" {budget} verified evaluations."
        ),
        output=_brief(ctx.state),
    )


def verify(ctx: Context, node_input: Candidate) -> Event:
    """Verifies and scores one proposed program and records the result."""
    used = ctx.state.get(_EVALUATIONS_KEY, 0) + 1
    ctx.state[_EVALUATIONS_KEY] = used
    history = list(ctx.state.get(_HISTORY_KEY, []))
    candidate_id = len(history)
    code_hash = _code_hash(node_input.code)
    seen = dict(ctx.state.get(_SEEN_KEY, {}))
    duplicate = seen.get(code_hash)
    if duplicate is not None:
        result: dict[str, Any] = {
            "valid": False,
            "score": None,
            "error": f"identical to candidate {duplicate}; propose a change",
        }
    else:
        result = verify_program(node_input.code)
    entry = _history_entry(
        candidate_id, node_input.idea, node_input.code, result
    )
    ctx.state[_HISTORY_KEY] = [*history, entry]
    if duplicate is None:
        ctx.state[_SEEN_KEY] = {**seen, code_hash: candidate_id}
    status = (
        f"${entry['monthly_cost']:,.2f}/month"
        if entry["valid"]
        else f"rejected ({entry['error']})"
    )
    return Event(
        message=f"Candidate {candidate_id} ({used}/{_max_evaluations()}): "
        f"{node_input.idea} -> {status}",
        output={
            **entry,
            "code": node_input.code,
            "score": result["score"],
        },
    )


def select(ctx: Context, node_input: dict[str, Any]) -> Event:
    """Updates the population and routes to another proposal or the report."""
    candidate = {
        "id": node_input["id"],
        "idea": node_input["idea"],
        "code": node_input["code"],
        "valid": node_input["valid"],
        "score": node_input["score"],
    }
    population = pop.add_candidate(
        ctx.state.get(_POPULATION_KEY, []), candidate, _population_size()
    )
    ctx.state[_POPULATION_KEY] = population
    best = pop.best(population)
    if best is None:
        raise RuntimeError("population is empty; init must seed the baseline")
    remaining = _max_evaluations() - ctx.state.get(_EVALUATIONS_KEY, 0)
    reached_bound = best["score"] <= problem.lower_bound()
    if remaining <= 0 or reached_bound:
        return Event(
            route=DONE,
            output={"best_id": best["id"], "reached_bound": reached_bound},
        )
    return Event(route=CONTINUE, output=_brief(ctx.state))


def build_report(state: Any) -> dict[str, Any]:
    """Re-verifies the baseline and the best program; returns the numbers."""
    best = pop.best(state.get(_POPULATION_KEY, []))
    if best is None:
        raise RuntimeError("No verified candidates; run init first.")
    baseline = verify_program(problem.BASELINE_PROGRAM)
    plan = verify_program(best["code"])
    if not plan["valid"]:
        raise RuntimeError(f"Best program no longer verifies: {plan['error']}")
    saved = round(baseline["monthly_cost"] - plan["monthly_cost"], 2)
    history = state.get(_HISTORY_KEY, [])
    return {
        "baseline_monthly_cost": baseline["monthly_cost"],
        "best_monthly_cost": plan["monthly_cost"],
        "monthly_savings": saved,
        "savings_percent": round(100 * saved / baseline["monthly_cost"], 1),
        "lower_bound_monthly_cost": _monthly(problem.lower_bound()),
        "evaluations_used": state.get(_EVALUATIONS_KEY, 0),
        "rejected": sum(1 for h in history if not h["valid"]),
        "clusters": [
            {
                "cluster": name,
                "baseline_nodes": baseline["per_instance"][name]["nodes"],
                "recommended_nodes": plan["per_instance"][name]["nodes"],
                "monthly_cost": _monthly(
                    plan["per_instance"][name]["hourly_cost"]
                ),
            }
            for name in plan["per_instance"]
        ],
        "tried": [
            {
                "id": h["id"],
                "idea": h["idea"],
                "monthly_cost": h["monthly_cost"],
                "error": h["error"],
            }
            for h in history
        ],
        "best_id": best["id"],
        "best_idea": best["idea"],
        "best_code": best["code"],
    }


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _mix(nodes: dict[str, int]) -> str:
    return ", ".join(f"{count}x `{name}`" for name, count in nodes.items())


def render_report(report: dict[str, Any]) -> str:
    """Formats the verifier's numbers as a Markdown report."""
    lines = [
        "## Kubernetes node cost plan",
        "",
        "| | Monthly cost |",
        "| --- | --- |",
        f"| Baseline: one `std-8` pool, first-fit | "
        f"${report['baseline_monthly_cost']:,.2f} |",
        f"| Best verified plan (candidate {report['best_id']}) | "
        f"**${report['best_monthly_cost']:,.2f}** |",
        f"| Savings | **${report['monthly_savings']:,.2f}"
        f" ({report['savings_percent']}%)** |",
        f"| Proven lower bound | ${report['lower_bound_monthly_cost']:,.2f} |",
        "",
        "### Recommended node mix",
        "",
        "| Cluster | Baseline nodes | Recommended nodes | Monthly cost |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {c['cluster']} | {_mix(c['baseline_nodes'])} | "
        f"{_mix(c['recommended_nodes'])} | ${c['monthly_cost']:,.2f} |"
        for c in report["clusters"]
    ]
    lines += [
        "",
        f"### Ideas tried ({report['evaluations_used']} evaluations,"
        f" {report['rejected']} rejected by the verifier)",
        "",
        "| # | Idea | Verified monthly cost |",
        "| --- | --- | --- |",
    ]
    for t in report["tried"]:
        cost = (
            f"${t['monthly_cost']:,.2f}"
            if t["monthly_cost"] is not None
            else f"rejected: {_cell(t['error'])[:120]}"
        )
        lines.append(f"| {t['id']} | {_cell(t['idea'])} | {cost} |")
    lines += [
        "",
        "### Best program",
        "",
        "```python",
        report["best_code"].rstrip(),
        "```",
    ]
    return "\n".join(lines)


def report(ctx: Context) -> Event:
    """Builds the savings report from verifier numbers only."""
    numbers = build_report(ctx.state)
    ctx.state[_REPORT_KEY] = render_report(numbers)
    return Event(
        message="All evaluations verified; writing the report.",
        output=_for_narrator(numbers),
    )


def _usd(value: float | None) -> str | None:
    return None if value is None else f"${value:,.2f}"


def _for_narrator(numbers: dict[str, Any]) -> dict[str, Any]:
    """The report without code, with money formatted as it appears above."""
    out = {k: v for k, v in numbers.items() if k != "best_code"}
    for key in (
        "baseline_monthly_cost",
        "best_monthly_cost",
        "monthly_savings",
        "lower_bound_monthly_cost",
    ):
        out[key] = _usd(numbers[key])
    out["savings_percent"] = f"{numbers['savings_percent']}%"
    out["clusters"] = [
        {**c, "monthly_cost": _usd(c["monthly_cost"])}
        for c in numbers["clusters"]
    ]
    out["tried"] = [
        {**t, "monthly_cost": _usd(t["monthly_cost"])} for t in numbers["tried"]
    ]
    return out


def deliver(ctx: Context, node_input: str) -> Event:
    """Sends one final answer: the verified report plus the summary.

    A2A clients such as Gemini Enterprise show the last message of a turn as
    the agent's answer, so the tables and the summary go out together.
    """
    text = f"{ctx.state[_REPORT_KEY]}\n\n{(node_input or '').strip()}"
    return Event(message=text.strip(), output=text.strip())


# ---------------------------------------------------------------------------
# LLM nodes
# ---------------------------------------------------------------------------

PROPOSER_INSTRUCTION = """\
You are the proposer in an AlphaEvolve-style loop that evolves a Kubernetes
workload placement program: which nodes to run and which pods go on each, so
every pod's CPU and memory requests fit and total node cost is as low as
possible. A deterministic verifier scores every program; you never do.

The input JSON gives you the problem statement, the node menu with prices,
each cluster's resource shape (memory per CPU core), the baseline and best
verified monthly cost, the proven lower bound, the best verified programs so
far (`parents`, cheapest first, with code), every idea tried so far with its
verified result (`tried`), the verifier's result for the latest candidate
(`last_feedback`), the evaluations remaining, and an optional
`operator_request` from the platform engineer.

Return ONE new, complete program that builds on the strongest parent(s) with
a clear hypothesis, for example: choose node types whose memory/CPU ratio
matches each cluster; sort pods by dominant resource share; best-fit on the
tighter dimension; mix node types and right-size the last, partly empty
nodes; local search that empties a node and re-places its pods. If
`last_feedback` has an error, fix its cause. Do not repeat an idea from
`tried` unchanged. Keep the program deterministic, standard-library only,
and about a second per call. Programs longer than `max_code_chars` are
rejected without running: check the parents' `code_chars`, and when a parent
is already large, replace or simplify parts of it instead of adding to it. Do not claim a
cost in `idea`; the verifier measures it.
"""

NARRATOR_INSTRUCTION = """\
You write the closing summary of a Kubernetes cost report for a platform
team. The input JSON is the verified report: baseline, best and lower-bound
monthly cost, savings, per-cluster node mix, and every idea tried.

Write at most 120 words of Markdown under the heading "### Summary":
what the recommended node mix changes in each cluster, why the winning idea
is cheaper (in terms of matching node shapes to workload shapes and packing),
and how close it is to the lower bound. Quote only numbers that appear in the
input, exactly as written but without quotation marks; never compute,
round or estimate a new number.
End by noting that the plan is a recommendation and nothing was changed in
the clusters. Do not repeat the tables; they are shown above your summary.
"""


def _model() -> Gemini:
    return Gemini(
        model=os.getenv("MODEL_NAME"),
        retry_options=types.HttpRetryOptions(attempts=3),
    )


def create_proposer() -> Agent:
    """The only creative step: writes one candidate program per round."""
    return Agent(
        name="propose",
        model=_model(),
        mode="single_turn",
        include_contents="none",
        description="Writes one new candidate placement program.",
        instruction=PROPOSER_INSTRUCTION,
        input_schema=ProposalBrief,
        output_schema=Candidate,
    )


def create_narrator() -> Agent:
    """Summarizes the verified report without introducing new numbers."""
    return Agent(
        name="narrate",
        model=_model(),
        mode="single_turn",
        include_contents="none",
        description="Summarizes the verified savings report.",
        instruction=NARRATOR_INSTRUCTION,
    )


def create_workflow(
    proposer: BaseNode | None = None,
    narrator: BaseNode | None = None,
) -> Workflow:
    """Builds the propose -> verify -> select graph.

    Args:
        proposer: Node named "propose" that turns a ProposalBrief into a
            Candidate. Defaults to the Gemini proposer; tests pass a stub.
        narrator: Node named "narrate" that summarizes the report. Defaults
            to the Gemini narrator; tests pass a stub.
    """
    propose = proposer or create_proposer()
    narrate = narrator or create_narrator()
    return Workflow(
        name="alphaevolve_optimizer",
        description=(
            "Cuts Kubernetes node cost: evolves a workload placement program"
            " with a propose -> verify -> select loop and reports verified"
            " monthly savings and the node mix per cluster."
        ),
        edges=[
            ("START", init, propose, verify, select),
            (select, {CONTINUE: propose, DONE: report}),
            (report, narrate, deliver),
        ],
    )


root_agent = create_workflow()

app = App(
    root_agent=root_agent,
    name="app",
)
