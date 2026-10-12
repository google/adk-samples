# Copyright 2025 Google LLC
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

"""Live eval harness: single-inference, 3-dimension scoring, vacuity guard.

Why not AgentEvaluator.evaluate(): its pytest aggregation only compares mean
metric scores and never inspects per-case final_eval_status — when every
inference fails there are no scores and the assert passes VACUOUSLY (Q19,
confirmed from source; see research/adk-eval-docs-context7.md). This harness
drives the same underlying eval stack but asserts inference status per case.

Installed API surface (verified against google-adk 2.5.0 source in
.venv/lib/python3.14/site-packages/google/adk/evaluation/, Phase 16
directive 4 — findings, with file paths):

(a) Eval set loading: ``load_eval_set_from_file(path, eval_set_id)`` in
    ``local_eval_sets_manager.py:176`` — validates the new ``EvalSet`` schema
    first, falls back to old-format conversion. All five eval sets under
    ``tests/integration/eval_sets/`` are already in the new schema.
(b) Inference: ``LocalEvalService.perform_inference(inference_request)``
    (``local_eval_service.py:143``) yields one ``InferenceResult`` per eval
    case with per-case ``status`` (``InferenceStatus.SUCCESS/FAILURE``) and
    ``error_message`` (``base_eval_service.py:137-172``) — inference
    exceptions are swallowed into FAILURE results by design
    (``local_eval_service.py:551-562``), which is exactly the behavior the
    vacuity guard exists to surface.
(c) Metric evaluation over existing inference results:
    ``LocalEvalService.evaluate(EvaluateRequest(inference_results=...,
    evaluate_config=EvaluateConfig(eval_metrics=...)))``
    (``local_eval_service.py:194``) — no re-inference; metrics come from
    ``get_eval_metrics_from_config(EvalConfig...)`` (``eval_config.py:229``).
    ``EvalConfig.criteria`` values parse as ``BaseCriterion`` with
    ``extra="allow"`` (``eval_metrics.py:109``); each evaluator re-validates
    into its own criterion type, e.g. ``ToolTrajectoryCriterion`` with
    string-coerced ``match_type`` (``trajectory_evaluator.py:78-96``,
    ``eval_metrics.py:252-261``).
(d) ``final_response_match_v2`` exists in installed 2.5.0
    (``eval_metrics.py:52`` PrebuiltMetrics, ``final_response_match_v2.py:130``
    ``FinalResponseMatchV2Evaluator`` with ``LlmAsAJudgeCriterion``), as does
    ``rubric_based_final_response_quality_v1`` (``eval_metrics.py:54``).
"""

import importlib
import json
import shutil
import sqlite3
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from app import config
from tests.conftest import MAIN_DB_PATH

try:
    from google.adk.evaluation.base_eval_service import (
        EvaluateConfig,
        EvaluateRequest,
        InferenceConfig,
        InferenceRequest,
        InferenceStatus,
    )
    from google.adk.evaluation.eval_case import Invocation, get_all_tool_calls
    from google.adk.evaluation.eval_config import (
        EvalConfig,
        get_eval_metrics_from_config,
    )
    from google.adk.evaluation.eval_metrics import EvalMetric
    from google.adk.evaluation.evaluator import EvalStatus
    from google.adk.evaluation.in_memory_eval_sets_manager import (
        InMemoryEvalSetsManager,
    )
    from google.adk.evaluation.local_eval_service import LocalEvalService
    from google.adk.evaluation.local_eval_sets_manager import (
        load_eval_set_from_file,
    )
except ImportError as e:
    # Fail loudly (ws01 lesson): a silent skip once hid a missing eval extra
    # behind "6 skipped". The ADK eval stack needs the [eval] extra deps
    # (pandas, tabulate, rouge-score, ...).
    raise ImportError(
        "google.adk.evaluation stack is unavailable. Install the eval extra:"
        " pip install 'google-adk[eval]' — see SETUP_INSTRUCTIONS.md (Tests)."
    ) from e

_APP_NAME = "live_eval"

EVAL_SETS_DIR = Path(__file__).parent / "eval_sets"
LIVE_EVAL_CONFIG_PATH = EVAL_SETS_DIR / "live_eval_config.json"


def get_eval_set_path(filename: str) -> str:
    """Absolute path to an eval_set JSON file under tests/integration/eval_sets."""
    return str(EVAL_SETS_DIR / filename)


@dataclass
class DimensionOutcome:
    score: float | None
    threshold: float
    passed: bool


@dataclass
class CaseOutcome:
    eval_id: str
    inference_ok: bool
    error_message: str | None
    actual_tool_calls: list[dict]
    final_response_text: str
    dimensions: dict[str, DimensionOutcome] = field(default_factory=dict)


_INFRA_MARKERS = (
    "credential",
    "permission denied",
    "permission_denied",
    "403",
    "quota",
    "resource_exhausted",
    "429",
    "unavailable",
    "503",
    "deadline",
    "connection",
    "getaddrinfo",
)  # kept from the old test_agents.py — infra breakage may xfail, never PASS

# Dimension name (matches live_eval_config.json "dimensions" key) eligible for
# the bounded one-shot re-judge [ws16 OWNER GATE 3, 2026-07-23].
_ANSWER_DIMENSION = "answer"

# Deterministic (non-LLM) dimensions: exact tool-name and tool-trajectory string
# matches. A failure here is AGENT nondeterminism (a different-but-not-wrong tool
# path this run), not judge noise — so it is handled by the GATE-4 bounded
# re-inference, never the GATE-3 answer re-judge. [ws16 OWNER GATE 4, 2026-07-23]
_DETERMINISTIC_DIMENSIONS = frozenset({"tools", "trajectory"})


def _deterministic_only_failure(
    dimensions: dict[str, "DimensionOutcome"],
) -> bool:
    """True iff the case has failures and EVERY failing dim is deterministic.

    Gates the GATE-4 re-inference: a case failing only on tools/trajectory
    (answer, if scored, is passing) is eligible for one end-to-end re-inference.
    A case whose answer dim is failing is NOT — that is the GATE-3 re-judge's
    job, and a fresh inference is not the right remedy for a borderline answer.
    """
    failed = [name for name, d in dimensions.items() if not d.passed]
    return bool(failed) and all(
        name in _DETERMINISTIC_DIMENSIONS for name in failed
    )


def _retry_outcome_or_keep_original(
    original: CaseOutcome, retry: CaseOutcome
) -> CaseOutcome:
    """[N1, ws16 final review] Retry replaces the original ONLY if it actually ran.

    The GATE-4 re-inference retries a case that hard-failed on a deterministic
    dim (tools/trajectory). If the retry's OWN inference dies (e.g. an infra
    hiccup — credentials/quota/connection), unconditionally replacing the
    original outcome with the failed retry would hand ``assert_eval_outcomes``
    a failed-inference case; if the error looks infra-shaped, the whole set
    xfails — discarding the original run's genuine tools/trajectory failure
    evidence behind an unrelated infra xfail. Keep the ORIGINAL outcome in
    that case so the set still FAILS on the real deterministic-dim mismatch.
    """
    return retry if retry.inference_ok else original


def load_dimension_metrics(
    config_path: str | Path = LIVE_EVAL_CONFIG_PATH,
) -> dict[str, list[EvalMetric]]:
    """Parse live_eval_config.json into per-dimension ADK EvalMetric lists.

    The file keeps each dimension's payload in NATIVE ADK ``EvalConfig``
    shape (``{"criteria": {...}}``); only the outer "dimensions" wrapper is
    ours (orchestration ours, formats native).
    """
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return {
        name: get_eval_metrics_from_config(EvalConfig.model_validate(dim))
        for name, dim in payload["dimensions"].items()
    }


def _load_root_agent():
    """Import the app's root agent (same convention as AgentEvaluator)."""
    return importlib.import_module("app.agent").root_agent


def _campaign_count(db_path) -> int:
    """Campaign-row count via a READ-ONLY connection (never touches the file).

    Read-only URI mode so the isolation guard itself can't mutate/`mtime`-touch
    the root DB it is protecting.
    """
    conn = sqlite3.connect(
        f"file:{Path(db_path).as_uri()[7:]}?mode=ro", uri=True
    )
    try:
        return conn.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]
    finally:
        conn.close()


def _assert_isolated_db() -> None:
    """Refuse to run eval inference against the real seeded DB.

    [ws16 GATE-3 isolation fix, 2026-07-23] Eval inference runs REAL tools that
    mutate the campaigns DB (create_campaign, ...). It must run against the
    live tier's isolated temp copy (``isolated_live_db`` patches
    ``config.DB_PATH``), never the root ``campaigns.db`` — otherwise
    eval-created campaigns leak into and ACCUMULATE in the real DB across runs
    (root-caused ws16: throwaway scripts that skipped the fixture did exactly
    this). This structural guard makes that leak impossible regardless of
    caller.
    """
    if Path(config.DB_PATH).resolve() == MAIN_DB_PATH.resolve():
        raise RuntimeError(
            "run_eval_set refuses to run against the root campaigns.db "
            f"({MAIN_DB_PATH}); point config.DB_PATH at an isolated copy first "
            "(the live tier's isolated_live_db fixture does this)."
        )


def _extract_actuals(
    invocations: list[Invocation],
) -> tuple[list[dict], str]:
    """Ordered actual tool calls (incl. transfer_to_agent) + final answer text."""
    tool_calls: list[dict] = []
    final_text = ""
    for invocation in invocations:
        for call in get_all_tool_calls(invocation.intermediate_data):
            tool_calls.append(
                {"name": call.name, "args": dict(call.args or {})}
            )
        if invocation.final_response and invocation.final_response.parts:
            texts = [p.text for p in invocation.final_response.parts if p.text]
            if texts:
                final_text = "\n".join(texts)
    return tool_calls, final_text


def _record_outcomes(
    outcomes: list[CaseOutcome], queries: dict[str, str], record_to: str
) -> None:
    """Dump actual trajectories + answers for calibration / trace export."""
    payload = [
        {
            "eval_id": o.eval_id,
            "query": queries.get(o.eval_id, ""),
            "inference_ok": o.inference_ok,
            "error_message": o.error_message,
            "actual_tool_calls": o.actual_tool_calls,
            "final_response_text": o.final_response_text,
            "dimensions": {
                name: {
                    "score": d.score,
                    "threshold": d.threshold,
                    "passed": d.passed,
                }
                for name, d in o.dimensions.items()
            },
        }
        for o in outcomes
    ]
    out_path = Path(record_to)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _outcome_from_result(result) -> CaseOutcome:
    """Build a CaseOutcome (inference status + actual trajectory) from a result."""
    tool_calls, final_text = _extract_actuals(result.inferences or [])
    return CaseOutcome(
        eval_id=result.eval_case_id,
        inference_ok=result.status == InferenceStatus.SUCCESS,
        error_message=result.error_message,
        actual_tool_calls=tool_calls,
        final_response_text=final_text,
    )


async def _infer_one_case(root_agent, eval_set_id: str, eval_case):
    """Run inference for ONE eval case against a FRESH seeded DB copy.

    [ws16 GATE-3 isolation fix] Resets config.DB_PATH to the seeded snapshot
    before the run so the case never sees another case's writes. Returns the
    single InferenceResult (or None if the stack yielded nothing).
    """
    shutil.copy2(MAIN_DB_PATH, config.DB_PATH)  # fresh seeded state per case
    manager = InMemoryEvalSetsManager()
    manager.create_eval_set(app_name=_APP_NAME, eval_set_id=eval_set_id)
    manager.add_eval_case(
        app_name=_APP_NAME, eval_set_id=eval_set_id, eval_case=eval_case
    )
    service = LocalEvalService(root_agent=root_agent, eval_sets_manager=manager)
    result = None
    async for inference_result in service.perform_inference(
        inference_request=InferenceRequest(
            app_name=_APP_NAME,
            eval_set_id=eval_set_id,
            inference_config=InferenceConfig(),
        )
    ):
        result = inference_result
    return result


async def _eval_one_dim(
    eval_service, inference_result, metrics
) -> DimensionOutcome:
    """Score a single dimension for a single inference result (no re-inference)."""
    request = EvaluateRequest(
        inference_results=[inference_result],
        evaluate_config=EvaluateConfig(eval_metrics=metrics),
    )
    outcome = DimensionOutcome(
        score=None, threshold=metrics[0].threshold or 0.0, passed=False
    )
    async for case_result in eval_service.evaluate(evaluate_request=request):
        metric_results = case_result.overall_eval_metric_results
        metric_result = metric_results[0] if metric_results else None
        outcome = DimensionOutcome(
            score=metric_result.score if metric_result else None,
            threshold=(
                metric_result.threshold
                if metric_result and metric_result.threshold is not None
                else (metrics[0].threshold or 0.0)
            ),
            passed=bool(
                metric_result and metric_result.eval_status == EvalStatus.PASSED
            ),
        )
    return outcome


async def _score_case(
    eval_service, inference_result, dimension_metrics
) -> dict[str, DimensionOutcome]:
    """Score all dims for one inference; bounded answer re-judge [ws16 GATE 3].

    final_response_match_v2 is an LLM judge and flakes on borderline-but-correct
    answers, so a failing ANSWER dim is re-judged ONCE against the SAME inference
    (no new agent run; the deterministic tools/trajectory dims are never
    re-judged here). A case fails the answer dim only if the judge fails it
    twice. Folding the re-judge in here means a GATE-4 re-inference's answer dim
    gets the same one-shot protection, so a recovered trajectory flake cannot be
    re-sunk by a fresh answer flake.
    """
    dims = {
        name: await _eval_one_dim(eval_service, inference_result, metrics)
        for name, metrics in dimension_metrics.items()
    }
    answer = dims.get(_ANSWER_DIMENSION)
    if (
        answer is not None
        and not answer.passed
        and dimension_metrics.get(_ANSWER_DIMENSION)
    ):
        eid = inference_result.eval_case_id
        msg = f"[eval-harness] answer-dimension retry (GATE 3) for: {eid}"
        print(msg)
        # [N2, ws16 final review] SETUP_INSTRUCTIONS.md's flakiness watchlist
        # promises every retry is "always visible" via print/warnings.warn — this
        # one previously only printed (silent on a passing run under pytest
        # capture). warn so the code matches that promise.
        warnings.warn(msg, stacklevel=2)
        redo = await _eval_one_dim(
            eval_service, inference_result, dimension_metrics[_ANSWER_DIMENSION]
        )
        print(
            f"[eval-harness] answer-retry {eid}: "
            + ("PASSED on retry" if redo.passed else "FAILED again")
        )
        dims[_ANSWER_DIMENSION] = redo
    return dims


async def run_eval_set(
    eval_set_path: str, *, record_to: str | None = None
) -> list[CaseOutcome]:
    """Run inference ONCE per case, then score all three dimensions.

    Inference results are reused across the per-dimension metric passes —
    only the judge metric makes additional (judge-model) LLM calls.
    """
    eval_set = load_eval_set_from_file(eval_set_path, Path(eval_set_path).stem)

    _assert_isolated_db()
    root_campaigns_before = _campaign_count(MAIN_DB_PATH)

    root_agent = _load_root_agent()

    # Full-set manager/service for the metric passes (evaluate + retry, which
    # do not touch the campaigns DB — they only run judge-model calls on the
    # recorded inference).
    eval_sets_manager = InMemoryEvalSetsManager()
    eval_sets_manager.create_eval_set(
        app_name=_APP_NAME, eval_set_id=eval_set.eval_set_id
    )
    for eval_case in eval_set.eval_cases:
        eval_sets_manager.add_eval_case(
            app_name=_APP_NAME,
            eval_set_id=eval_set.eval_set_id,
            eval_case=eval_case,
        )
    eval_service = LocalEvalService(
        root_agent=root_agent, eval_sets_manager=eval_sets_manager
    )

    # 1. Inference + scoring — ONE case at a time, each against a FRESH seeded
    # DB copy. [ws16 GATE-3 isolation fix] ADK's perform_inference runs a set's
    # cases CONCURRENTLY (asyncio.Semaphore + as_completed) against the single
    # shared config.DB_PATH; a mutating case (create-campaign) then races with
    # and pollutes a read case (get-campaign-locations, whose reference lists
    # exactly the 4 seeded stores). Per-case reset gives true isolation — no
    # case ever sees another case's writes. Each successful inference is scored
    # across all dims (with the GATE-3 bounded answer re-judge folded in).
    dimension_metrics = load_dimension_metrics()
    outcomes_by_id: dict[str, CaseOutcome] = {}
    for eval_case in eval_set.eval_cases:
        result = await _infer_one_case(
            root_agent, eval_set.eval_set_id, eval_case
        )
        outcome = _outcome_from_result(result)
        if result.status == InferenceStatus.SUCCESS:
            outcome.dimensions = await _score_case(
                eval_service, result, dimension_metrics
            )
        outcomes_by_id[result.eval_case_id] = outcome

    # 2. Bounded re-inference for deterministic-dim flakes [ws16 OWNER GATE 4,
    # 2026-07-23]. tools/trajectory are exact string matches, so a failure there
    # is AGENT nondeterminism (a different-but-not-wrong tool path this run), not
    # judge noise — the GATE-3 answer re-judge cannot address it. Re-run the
    # single case's inference ONCE end-to-end (fresh seeded DB) and re-score
    # every dim; the case fails only if it fails twice. Every trigger is logged
    # AND warned (owner wants trajectory nondeterminism to stay observable).
    cases_by_id = {c.eval_id: c for c in eval_set.eval_cases}
    for eval_case in eval_set.eval_cases:
        eid = eval_case.eval_id
        outcome = outcomes_by_id.get(eid)
        if (
            outcome is None
            or not outcome.inference_ok
            or not _deterministic_only_failure(outcome.dimensions)
        ):
            continue
        failed = ", ".join(
            sorted(n for n, d in outcome.dimensions.items() if not d.passed)
        )
        msg = (
            f"deterministic-dim failure on {eid} ({failed}) — re-running "
            "inference ONCE [ws16 OWNER GATE 4, 2026-07-23]"
        )
        print(f"[eval-harness] WARNING: {msg}")
        warnings.warn(msg, stacklevel=2)
        result = await _infer_one_case(
            root_agent, eval_set.eval_set_id, cases_by_id[eid]
        )
        retry = _outcome_from_result(result)
        if result.status == InferenceStatus.SUCCESS:
            retry.dimensions = await _score_case(
                eval_service, result, dimension_metrics
            )
        final = _retry_outcome_or_keep_original(outcome, retry)
        outcomes_by_id[eid] = final
        if final is not retry:
            # [N1, ws16 final review] retry's own inference died — original
            # deterministic-dim failure preserved, not infra-masked.
            print(
                f"[eval-harness] re-inference {eid}: retry inference itself "
                f"failed ({retry.error_message}) — keeping ORIGINAL outcome "
                "(not infra-masking the first-run failure)"
            )
        else:
            still_failed = sorted(
                n for n, d in retry.dimensions.items() if not d.passed
            )
            if not still_failed:
                print(f"[eval-harness] re-inference {eid}: PASSED on retry")
            else:
                print(
                    f"[eval-harness] re-inference {eid}: FAILED again "
                    f"({', '.join(still_failed)})"
                )

    # Stable order: as authored in the eval set.
    outcomes = [
        outcomes_by_id[c.eval_id]
        for c in eval_set.eval_cases
        if c.eval_id in outcomes_by_id
    ]

    if record_to:
        queries = {
            c.eval_id: (
                c.conversation[0].user_content.parts[0].text or ""
                if c.conversation and c.conversation[0].user_content.parts
                else ""
            )
            for c in eval_set.eval_cases
        }
        _record_outcomes(outcomes, queries, record_to)

    # Post-run: the root DB must be byte-for-byte untouched. All inference ran
    # against config.DB_PATH (an isolated temp copy, enforced by
    # _assert_isolated_db above); the root count must be exactly what it was.
    # [ws16 GATE-3 isolation fix, 2026-07-23]
    root_campaigns_after = _campaign_count(MAIN_DB_PATH)
    if root_campaigns_after != root_campaigns_before:
        raise RuntimeError(
            "eval run mutated the root campaigns.db: "
            f"{root_campaigns_before} -> {root_campaigns_after} campaigns; "
            "isolation broke."
        )

    return outcomes


def assert_eval_outcomes(
    outcomes: list[CaseOutcome], *, expect_cases: int
) -> None:
    """Vacuity guard + per-dimension gate (Q19).

    Zero or partial inference runs must never PASS; infra breakage may
    xfail, never PASS; any failed dimension fails the test.
    """
    import pytest

    if len(outcomes) != expect_cases:
        pytest.fail(
            f"expected {expect_cases} eval cases, only {len(outcomes)} produced"
            " results — partial/zero runs must not pass"
        )
    failed_inference = [o for o in outcomes if not o.inference_ok]
    if failed_inference:
        msgs = "; ".join(
            f"{o.eval_id}: {o.error_message}" for o in failed_inference
        )
        if all(
            any(m in (o.error_message or "").lower() for m in _INFRA_MARKERS)
            for o in failed_inference
        ):
            pytest.xfail(f"integration infrastructure unavailable: {msgs}")
        pytest.fail(f"inference failed (NOT infra): {msgs}")
    dim_failures = [
        f"{o.eval_id}[{name}]: score={d.score} < {d.threshold}"
        for o in outcomes
        for name, d in o.dimensions.items()
        if not d.passed
    ]
    if dim_failures:
        pytest.fail("eval dimension failures:\n  " + "\n  ".join(dim_failures))
