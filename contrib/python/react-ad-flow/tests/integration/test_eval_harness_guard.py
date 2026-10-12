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

"""Deliberate-break checks: zero/partial inference must never PASS (Q19)."""

import pytest
from _pytest.outcomes import Failed, XFailed

from tests.integration.eval_harness import (
    CaseOutcome,
    DimensionOutcome,
    assert_eval_outcomes,
)

pytestmark = pytest.mark.integration


def _case(eval_id="c1", ok=True, err=None):
    return CaseOutcome(
        eval_id=eval_id,
        inference_ok=ok,
        error_message=err,
        actual_tool_calls=[],
        final_response_text="",
    )


def test_all_inference_failed_never_passes():
    bad = [_case(ok=False, err="something exploded (not infra-shaped)")]
    with pytest.raises(Failed):
        assert_eval_outcomes(bad, expect_cases=1)


def test_infra_failure_xfails_not_passes():
    bad = [_case(ok=False, err="403 PERMISSION_DENIED on aiplatform")]
    with pytest.raises(XFailed):
        assert_eval_outcomes(bad, expect_cases=1)


def test_partial_results_never_pass():
    with pytest.raises(Failed):
        assert_eval_outcomes([_case()], expect_cases=4)


def test_failed_dimension_fails():
    good = _case()
    good.dimensions["tools"] = DimensionOutcome(
        score=0.0, threshold=1.0, passed=False
    )
    with pytest.raises(Failed, match=r"c1\[tools\]"):
        assert_eval_outcomes([good], expect_cases=1)


def test_all_green_passes():
    good = _case()
    good.dimensions["tools"] = DimensionOutcome(
        score=1.0, threshold=1.0, passed=True
    )
    assert_eval_outcomes([good], expect_cases=1)


def test_answer_retry_targets_only_the_answer_dimension():
    """GATE-3 bounded retry re-judges ONLY the answer dimension.

    Guards the retry constant against config drift: if the config's answer
    dimension were renamed, the retry would silently no-op. The deterministic
    tools/trajectory dimensions exist and are never retried. Pure (no live).
    """
    from tests.integration.eval_harness import (
        _ANSWER_DIMENSION,
        load_dimension_metrics,
    )

    dims = load_dimension_metrics()
    assert _ANSWER_DIMENSION == "answer"
    assert _ANSWER_DIMENSION in dims, (
        "retry would no-op: no 'answer' dimension in config"
    )
    assert {"tools", "trajectory"} <= set(
        dims
    )  # deterministic dims, not retried


def test_isolation_guard_refuses_root_db(monkeypatch):
    """The isolation guard must REFUSE to run against the root campaigns.db.

    [ws16 GATE-3 isolation fix] Regression guard for the eval-set DB leak:
    eval inference mutates the campaigns DB via real tools, so pointing it at
    the seeded root DB would accumulate eval-created campaigns across runs.
    Pure (no live inference) — just asserts the structural guard fires.
    """
    from app import config
    from tests.conftest import MAIN_DB_PATH
    from tests.integration.eval_harness import _assert_isolated_db

    monkeypatch.setattr(config, "DB_PATH", str(MAIN_DB_PATH))
    with pytest.raises(RuntimeError, match=r"root campaigns\.db"):
        _assert_isolated_db()


def test_isolation_guard_allows_temp_db(monkeypatch, tmp_path):
    """The isolation guard passes when config.DB_PATH is an isolated copy."""
    from app import config
    from tests.integration.eval_harness import _assert_isolated_db

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "campaigns.db"))
    _assert_isolated_db()  # must not raise


# --- GATE-4 bounded eval re-inference: selection predicate (pure, no live) ---


def _dim(passed):
    return DimensionOutcome(
        score=1.0 if passed else 0.0, threshold=1.0, passed=passed
    )


def test_deterministic_only_failure_predicate():
    """[ws16 OWNER GATE 4] Re-inference fires ONLY for deterministic-dim flakes.

    A case failing on tools/trajectory (answer passing) is eligible; a case
    whose answer dim fails is NOT (that is GATE-3's re-judge). Pure — no live.
    """
    from tests.integration.eval_harness import (
        _DETERMINISTIC_DIMENSIONS,
        _deterministic_only_failure,
    )

    assert _DETERMINISTIC_DIMENSIONS == frozenset({"tools", "trajectory"})
    assert "answer" not in _DETERMINISTIC_DIMENSIONS

    # trajectory-only failure -> eligible
    assert _deterministic_only_failure(
        {"tools": _dim(True), "trajectory": _dim(False), "answer": _dim(True)}
    )
    # tools + trajectory both fail -> eligible
    assert _deterministic_only_failure(
        {"tools": _dim(False), "trajectory": _dim(False), "answer": _dim(True)}
    )
    # answer-only failure -> NOT eligible (GATE-3 territory)
    assert not _deterministic_only_failure(
        {"tools": _dim(True), "trajectory": _dim(True), "answer": _dim(False)}
    )
    # trajectory AND answer fail -> NOT eligible (answer is failing)
    assert not _deterministic_only_failure(
        {"tools": _dim(True), "trajectory": _dim(False), "answer": _dim(False)}
    )
    # all pass -> not eligible
    assert not _deterministic_only_failure(
        {"tools": _dim(True), "trajectory": _dim(True), "answer": _dim(True)}
    )


def test_retry_keeps_original_when_retry_inference_itself_fails():
    """[N1, ws16 final review] A retry whose OWN inference dies must NOT mask
    the original deterministic-dim failure behind an infra xfail.

    If ``_retry_outcome_or_keep_original`` unconditionally replaced the
    original with a failed-inference retry, the set would report a failed
    (possibly infra-shaped) inference instead of the genuine tools/trajectory
    mismatch it originally caught — turning a real bug into a silent xfail.
    The original outcome must be preserved so the set FAILS, not xfails.
    """
    from tests.integration.eval_harness import _retry_outcome_or_keep_original

    original = _case(ok=True)
    original.dimensions["trajectory"] = _dim(
        False
    )  # genuine deterministic-dim failure

    failed_retry = _case(
        ok=False, err="503 UNAVAILABLE (infra hiccup on retry)"
    )

    kept = _retry_outcome_or_keep_original(original, failed_retry)
    assert kept is original
    assert kept.dimensions["trajectory"].passed is False

    # End-to-end: assert_eval_outcomes must FAIL (not xfail) on the preserved
    # outcome — the deterministic-dim mismatch stays visible as a real failure.
    with pytest.raises(Failed):
        assert_eval_outcomes([kept], expect_cases=1)


def test_retry_replaces_original_when_retry_inference_succeeds():
    """A retry that actually ran (inference_ok) IS authoritative and replaces
    the original — the GATE-4 contract for a genuine retry outcome.
    """
    from tests.integration.eval_harness import _retry_outcome_or_keep_original

    original = _case(ok=True)
    original.dimensions["trajectory"] = _dim(False)

    successful_retry = _case(ok=True)
    successful_retry.dimensions["trajectory"] = _dim(True)

    kept = _retry_outcome_or_keep_original(original, successful_retry)
    assert kept is successful_retry
    assert kept.dimensions["trajectory"].passed is True


# --- GATE-4 bounded media re-judge: retry logic (pure, fake judge, no live) ---


def _verdict(check, verdict_str, severity, evidence="e"):
    return {
        check: {
            "verdict": verdict_str,
            "severity": severity,
            "evidence": evidence,
        }
    }


def test_media_retry_recovers_on_second_pass():
    """[ws16 OWNER GATE 4] A hard-fail that clears on re-judge yields a pass."""
    from tests.live.judge import judge_with_hard_retry

    calls = {"n": 0}

    def fake_judge():
        calls["n"] += 1
        if calls["n"] == 1:
            return _verdict("no_rendered_text", "fail", "hard", "hallucinated")
        return _verdict("no_rendered_text", "pass", "hard", "clean")

    verdict = judge_with_hard_retry("label", fake_judge)
    assert calls["n"] == 2  # re-judged exactly once
    assert verdict["no_rendered_text"]["verdict"] == "pass"


def test_media_retry_fails_twice_for_real_violation():
    """A genuine hard violation must hard-fail BOTH times (negative-control shape)."""
    from tests.live.judge import hard_failed_checks, judge_with_hard_retry

    calls = {"n": 0}

    def fake_judge():
        calls["n"] += 1
        return _verdict("no_rendered_text", "fail", "hard", "real banner")

    verdict = judge_with_hard_retry("label", fake_judge)
    assert calls["n"] == 2
    assert hard_failed_checks(verdict) == ["no_rendered_text"]


def test_media_retry_skips_when_only_warn_fails():
    """A warn-only failure is not a hard failure — no re-judge, single call."""
    from tests.live.judge import hard_failed_checks, judge_with_hard_retry

    calls = {"n": 0}

    def fake_judge():
        calls["n"] += 1
        return {
            "no_rendered_text": {
                "verdict": "pass",
                "severity": "hard",
                "evidence": "clean",
            },
            "setting_mood_plausible": {
                "verdict": "fail",
                "severity": "warn",
                "evidence": "meh",
            },
        }

    verdict = judge_with_hard_retry("label", fake_judge)
    assert calls["n"] == 1  # no retry for a warn-only failure
    assert hard_failed_checks(verdict) == []
