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
"""An AlphaEvolve-style evolutionary code optimizer built with ADK.

The loop is: propose -> verify -> select.

- Propose: the Gemini agent writes a new candidate program, using the best
  programs found so far (and their scores) as parents.
- Verify: `evaluate_candidate` runs the candidate in a resource-limited
  subprocess and scores its outputs deterministically in this process.
- Select: the best verified candidates are kept in session state and fed back
  to the model for the next round, until the evaluation budget is spent or
  the known lower bound is reached.
"""

import os
from typing import Any

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools import ToolContext
from google.genai import types

from . import population as pop
from . import problem
from .sandbox import run_candidate

_POPULATION_KEY = "evolve_population"
_EVALUATIONS_KEY = "evolve_evaluations"
_NEXT_ID_KEY = "evolve_next_id"


def _max_evaluations() -> int:
    return int(os.getenv("MAX_EVALUATIONS", "10"))


def _population_size() -> int:
    return int(os.getenv("POPULATION_SIZE", "4"))


def _timeout_seconds() -> float:
    return float(os.getenv("CANDIDATE_TIMEOUT_SECONDS", "10"))


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


def _register(
    tool_context: ToolContext, code: str, idea: str, result: dict[str, Any]
) -> dict[str, Any]:
    """Assigns an id, updates the population in session state."""
    candidate_id = tool_context.state.get(_NEXT_ID_KEY, 0)
    tool_context.state[_NEXT_ID_KEY] = candidate_id + 1
    candidate = {
        "id": candidate_id,
        "idea": idea,
        "code": code,
        "valid": result["valid"],
        "score": result["score"],
    }
    tool_context.state[_POPULATION_KEY] = pop.add_candidate(
        tool_context.state.get(_POPULATION_KEY, []),
        candidate,
        _population_size(),
    )
    return candidate


def get_problem(tool_context: ToolContext) -> dict[str, Any]:
    """Returns the optimization problem and seeds the population with a baseline.

    Call this once at the start of an optimization run.

    Returns:
        The problem statement, the baseline program and its verified score,
        the lower bound on the score, and the evaluation budget.
    """
    baseline = verify_program(problem.BASELINE_PROGRAM)
    if not tool_context.state.get(_POPULATION_KEY):
        _register(
            tool_context,
            problem.BASELINE_PROGRAM,
            "baseline: first-fit",
            baseline,
        )
        tool_context.state[_EVALUATIONS_KEY] = 0
    best = pop.best(tool_context.state[_POPULATION_KEY])
    return {
        "problem": problem.PROBLEM_STATEMENT,
        "baseline_program": problem.BASELINE_PROGRAM,
        "baseline_score": baseline["score"],
        "baseline_per_instance": baseline.get("per_instance"),
        "best_score": best["score"] if best else None,
        "lower_bound": problem.lower_bound(),
        "evaluation_budget": _max_evaluations(),
        "evaluations_used": tool_context.state.get(_EVALUATIONS_KEY, 0),
    }


def get_parents(tool_context: ToolContext) -> dict[str, Any]:
    """Returns the best programs found so far, with code, to build on.

    Returns:
        The current population (best first), each with id, score, idea and code.
    """
    population = sorted(
        tool_context.state.get(_POPULATION_KEY, []),
        key=lambda c: (c["score"], c["id"]),
    )
    return {
        "parents": [
            {
                "id": c["id"],
                "score": c["score"],
                "idea": c["idea"],
                "code": c["code"],
            }
            for c in population
        ],
        "lower_bound": problem.lower_bound(),
    }


def evaluate_candidate(
    code: str, idea: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Verifies and scores a candidate program, and updates the population.

    Args:
        code: Complete Python source defining `pack(items, capacity)`.
        idea: One sentence describing what changed versus the parent.

    Returns:
        The candidate's validity, score (lower is better) or error, whether it
        is a new best, the current best score, and the remaining budget.
    """
    used = tool_context.state.get(_EVALUATIONS_KEY, 0)
    if used >= _max_evaluations():
        return {
            "budget_exhausted": True,
            "message": "No evaluations left. Report results.",
        }
    tool_context.state[_EVALUATIONS_KEY] = used + 1

    previous_best = pop.best(tool_context.state.get(_POPULATION_KEY, []))
    result = verify_program(code)
    candidate = _register(tool_context, code, idea, result)
    current_best = pop.best(tool_context.state[_POPULATION_KEY])
    remaining = _max_evaluations() - (used + 1)
    reached_bound = (
        current_best is not None
        and current_best["score"] <= problem.lower_bound()
    )
    return {
        "candidate_id": candidate["id"],
        "valid": result["valid"],
        "score": result["score"],
        "per_instance": result.get("per_instance"),
        "error": result.get("error"),
        "is_new_best": bool(
            result["valid"]
            and (
                previous_best is None
                or result["score"] < previous_best["score"]
            )
        ),
        "best_score": current_best["score"] if current_best else None,
        "lower_bound": problem.lower_bound(),
        "evaluations_remaining": remaining,
        "done": remaining <= 0 or reached_bound,
        "population": pop.summarize(tool_context.state[_POPULATION_KEY]),
    }


INSTRUCTION = """\
You are an evolutionary code optimizer in the style of AlphaEvolve. You improve
a program by repeatedly proposing a candidate, verifying it with a tool, and
building on the best verified results.

When the user asks you to start (or to optimize / evolve / run):
1. Call `get_problem` once. Read the problem, the baseline and its score.
2. Repeat until a tool result has `done: true` or `budget_exhausted: true`:
   a. Call `get_parents` and pick one or two strong parents.
   b. Propose ONE new complete program. Make a focused change with a clear
      hypothesis (e.g. sort order, best-fit vs first-fit, local search to
      empty the fullest-but-one bin, pairing complementary sizes). Keep it
      deterministic and standard-library only.
   c. Call `evaluate_candidate` with the full code and a one-sentence `idea`.
   d. Use the score, per-instance results and any error to decide the next
      change. If a candidate is invalid, fix the cause rather than retrying
      blindly. Do not resubmit identical code.
3. Finish with a short report: baseline score, best score, lower bound, a
   table of the ideas you tried with their scores, and the best program in a
   Python code block.

Never claim a score that did not come from `evaluate_candidate`. Lower scores
are better. Do not ask the user for confirmation between iterations.
"""


def create_agent() -> Agent:
    """Creates a fresh, isolated instance of the Agent."""
    return Agent(
        name="root_agent",
        model=Gemini(
            model=os.getenv("MODEL_NAME"),
            retry_options=types.HttpRetryOptions(attempts=3),
        ),
        description="Evolves programs with a propose -> verify -> select loop.",
        instruction=INSTRUCTION,
        tools=[get_problem, get_parents, evaluate_candidate],
    )


root_agent = create_agent()

app = App(
    root_agent=root_agent,
    name="app",
)
