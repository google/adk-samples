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
"""Unit tests for the scanner, validator, workspace and workflow. No model calls.

The workflow tests run the real ADK 2.0 graph with the Gemini patch agent
swapped for a deterministic function node that returns reference fixes.
"""

import asyncio
import hashlib
import os
import shutil
from pathlib import Path

import pytest
from google.adk import Workflow
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types
from reference_fixes import (
    ALL_FIXES,
    DB_BROKEN,
    DB_FIX,
    INVOICES_FIX,
    INVOICES_INCOMPLETE,
)

from app import agent, config, scanner, validator, workspace

TIMEOUT = 60
SKIPS = os.environ["BANDIT_SKIPS"]  # loaded from .env.example by conftest


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for rel in workspace.list_files(root, suffix=""):
        if "__pycache__" in rel:
            continue
        digest.update(rel.encode())
        digest.update((root / rel).read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def baseline() -> dict:
    return scanner.scan(str(workspace.BUNDLED_TARGET), TIMEOUT, None, SKIPS)


@pytest.fixture
def work(tmp_path: Path) -> Path:
    return workspace.copy_tree(workspace.BUNDLED_TARGET, str(tmp_path))


@pytest.fixture
def state(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.delenv("TARGET_DIR", raising=False)
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.setenv("VALIDATION_TIMEOUT_SECONDS", str(TIMEOUT))
    monkeypatch.setenv("MAX_PATCH_ATTEMPTS", "8")
    return {}


# --- scanner -----------------------------------------------------------------


def test_scanner_finds_the_sample_vulnerabilities(baseline: dict) -> None:
    keys = set(scanner.finding_counts(baseline))
    assert {
        "orders/db.py:B608",
        "orders/cart.py:B301",
        "orders/invoices.py:B602",
        "orders/auth.py:B105",
    } <= keys
    assert not any(k.startswith("tests/") for k in keys)
    assert baseline["scanner_error"] is None
    assert baseline["tests_error"] is None


def test_test_harness_reports_per_test_outcomes(baseline: dict) -> None:
    failed = scanner.failed_tests(baseline)
    passed = scanner.passed_tests(baseline)
    assert len(passed) == 9  # every functional test passes
    assert len(failed) == 8  # every exploit test fails
    assert all(".test_security." in t for t in failed)
    assert "tests.test_orders.CartTest.test_cart_round_trip" in passed


def test_generic_test_command_parsing() -> None:
    run = {
        "returncode": 1,
        "stdout": "PASSED tests/test_a.py::test_ok\n"
        "FAILED tests/test_a.py::test_bad - AssertionError: boom\n",
        "stderr": "",
        "timed_out": False,
    }
    tests = scanner._parse_generic(run)["tests"]
    assert tests == [
        {"id": "tests/test_a.py::test_ok", "status": "passed", "detail": ""},
        {
            "id": "tests/test_a.py::test_bad",
            "status": "failed",
            "detail": "AssertionError: boom",
        },
    ]
    fallback = scanner._parse_generic({**run, "stdout": "1 failed\n"})
    assert fallback["tests"][0]["id"] == "test-suite"
    assert fallback["tests"][0]["status"] == "failed"
    argv = scanner.build_test_argv("python -m pytest -rA -q")
    assert argv[0] != "python" and argv[1:] == ["-m", "pytest", "-rA", "-q"]


# --- validator ---------------------------------------------------------------


def test_validator_accepts_a_correct_fix(work: Path, baseline: dict) -> None:
    verdict = validator.validate_patch(
        work, "orders/db.py", DB_FIX, baseline, TIMEOUT, None, SKIPS
    )
    assert verdict["accepted"], verdict["reasons"]
    assert verdict["resolved_findings"] == ["orders/db.py:B608"]
    assert len(verdict["fixed_tests"]) == 3
    assert verdict["regressions"] == []
    # Validation happens in a scratch copy; the working copy is untouched.
    assert 'f"SELECT' in (work / "orders/db.py").read_text()


def test_validator_rejects_a_fix_that_breaks_behaviour(
    work: Path, baseline: dict
) -> None:
    verdict = validator.validate_patch(
        work, "orders/db.py", DB_BROKEN, baseline, TIMEOUT, None, SKIPS
    )
    assert not verdict["accepted"]
    assert verdict["regressions"] == [
        "tests.test_orders.OrdersTest.test_find_orders_by_status"
    ]


def test_validator_rejects_an_incomplete_fix(
    work: Path, baseline: dict
) -> None:
    verdict = validator.validate_patch(
        work,
        "orders/invoices.py",
        INVOICES_INCOMPLETE,
        baseline,
        TIMEOUT,
        None,
        SKIPS,
    )
    assert not verdict["accepted"]
    assert verdict["fixed_tests"] == []
    assert any("no finding resolved" in r for r in verdict["reasons"])


@pytest.mark.parametrize(
    ("rel_path", "content", "reason"),
    [
        ("tests/test_security.py", "pass\n", "test files"),
        ("orders/db.py", "def broken(:\n", "syntax error"),
        (
            "orders/db.py",
            DB_FIX.replace("import sqlite3", "import sqlite3  # nosec"),
            "nosec",
        ),
        ("../outside.py", "x = 1\n", "relative"),
        ("orders/missing.py", "x = 1\n", "not found"),
    ],
)
def test_validator_prechecks(
    work: Path, baseline: dict, rel_path: str, content: str, reason: str
) -> None:
    verdict = validator.validate_patch(
        work, rel_path, content, baseline, TIMEOUT, None, SKIPS
    )
    assert not verdict["accepted"]
    assert verdict["after"] is None  # rejected before running anything
    assert reason in verdict["reasons"][0]


# --- workspace ---------------------------------------------------------------


def test_safe_path_and_test_detection(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n")
    (tmp_path / "link.py").symlink_to(tmp_path / "pkg" / "mod.py")
    assert workspace.safe_path(tmp_path, "pkg/mod.py").is_file()
    for bad in ("../etc/passwd", "/etc/passwd", "link.py", ""):
        with pytest.raises(workspace.WorkspaceError):
            workspace.safe_path(tmp_path, bad)
    assert workspace.is_test_path("tests/test_x.py")
    assert workspace.is_test_path("pkg/conftest.py")
    assert workspace.is_test_path("pkg/thing_test.py")
    assert not workspace.is_test_path("orders/db.py")


def test_working_copy_replays_accepted_patches() -> None:
    copy = workspace.create_working_copy(
        workspace.BUNDLED_TARGET, {"orders/db.py": DB_FIX}
    )
    try:
        assert (copy / "orders/db.py").read_text() == DB_FIX
        assert copy.resolve() != workspace.BUNDLED_TARGET
    finally:
        shutil.rmtree(copy.parent)


# --- deterministic workflow steps ---------------------------------------------


def test_start_run_records_the_baseline(state: dict) -> None:
    result = agent.start_run(state)
    assert result["route"] == "has_findings"
    assert result["summary"]["tests_passed"] == 9
    assert state["cm_baseline"] is state["cm_current"]
    assert state["cm_attempts"] == []
    assert Path(state["cm_work_dir"]) != workspace.BUNDLED_TARGET


def test_start_run_routes_clean_targets_to_report(
    state: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "already-fixed"
    shutil.copytree(workspace.BUNDLED_TARGET, repo)
    for rel, content in ALL_FIXES.items():
        (repo / rel).write_text(content)
    monkeypatch.setenv("TARGET_DIR", str(repo))
    assert agent.start_run(state)["route"] == "clean"
    assert state["cm_source_dir"] == str(repo)


def test_start_run_reports_an_unusable_target(
    state: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TARGET_DIR", str(tmp_path / "missing"))
    result = agent.start_run(state)
    assert result["route"] == "clean"
    assert "not found" in result["error"]
    assert "not found" in agent.render_report(agent.build_report(state))


def test_patch_request_carries_evidence_and_feedback(state: dict) -> None:
    agent.start_run(state)
    state["cm_feedback"] = "rejected: regression"
    request = agent.PatchRequest.model_validate(
        agent.build_patch_request(state)
    )
    assert "orders/db.py" in request.source_files
    assert "tests/test_security.py" in request.test_files
    assert not any(workspace.is_test_path(f) for f in request.source_files)
    assert len(request.failing_tests) == 8
    assert request.previous_attempt_feedback == "rejected: regression"
    assert request.attempts_remaining == 8


def test_apply_proposal_routes_and_records_feedback(state: dict) -> None:
    agent.start_run(state)
    rejected = agent.apply_proposal(
        state,
        {
            "file_path": "orders/invoices.py",
            "new_content": INVOICES_INCOMPLETE,
            "rationale": "strip ../",
        },
    )
    assert not rejected["accepted"]
    assert rejected["route"] == "continue"
    assert "no finding resolved" in state["cm_feedback"]
    accepted = agent.apply_proposal(
        state,
        agent.PatchProposal(
            file_path="orders/invoices.py",
            new_content=INVOICES_FIX,
            rationale="contain paths; no shell",
        ),
    )
    assert accepted["accepted"]
    assert accepted["route"] == "continue"
    assert state["cm_feedback"] == ""
    assert state["cm_applied"] == {"orders/invoices.py": INVOICES_FIX}
    missing = agent.apply_proposal(state, None)
    assert not missing["accepted"]
    assert "PatchProposal" in missing["reasons"][0]
    assert len(state["cm_attempts"]) == 3


def test_budget_ends_the_loop(
    state: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_PATCH_ATTEMPTS", "1")
    agent.start_run(state)
    result = agent.apply_proposal(
        state,
        {
            "file_path": "orders/db.py",
            "new_content": DB_BROKEN,
            "rationale": "",
        },
    )
    assert result["attempts_remaining"] == 0
    assert result["route"] == "done"


def test_missing_config_fails_with_a_clear_error(
    state: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent.start_run(state)
    monkeypatch.delenv("MAX_PATCH_ATTEMPTS")
    with pytest.raises(config.ConfigError, match="MAX_PATCH_ATTEMPTS"):
        agent.apply_proposal(
            state,
            {
                "file_path": "orders/db.py",
                "new_content": DB_FIX,
                "rationale": "",
            },
        )


# --- the ADK 2.0 graph, end to end ----------------------------------------------


def _scripted_patcher(proposals: list[tuple[str, str]]):
    """A function node standing in for the Gemini patch agent."""
    queue = list(proposals)

    def scripted_patcher(node_input: dict) -> dict:
        agent.PatchRequest.model_validate(node_input)
        rel, content = queue.pop(0)
        return {"file_path": rel, "new_content": content, "rationale": "ref"}

    return scripted_patcher


def _run(workflow: Workflow) -> tuple[list[str], dict]:
    async def go() -> tuple[list[str], dict]:
        runner = InMemoryRunner(app=App(root_agent=workflow, name="app"))
        session = await runner.session_service.create_session(
            app_name="app", user_id="u"
        )
        texts = []
        message = types.Content(role="user", parts=[types.Part(text="fix it")])
        async for event in runner.run_async(
            user_id="u", session_id=session.id, new_message=message
        ):
            for part in (event.content.parts if event.content else None) or []:
                if part.text:
                    texts.append(part.text)
        final = await runner.session_service.get_session(
            app_name="app", user_id="u", session_id=session.id
        )
        return texts, final.state

    return asyncio.run(go())


def test_workflow_fixes_everything_and_reports(state: dict) -> None:
    before_hash = _tree_hash(workspace.BUNDLED_TARGET)
    proposals = [
        ("orders/invoices.py", INVOICES_INCOMPLETE),
        *ALL_FIXES.items(),
    ]
    texts, final_state = _run(
        agent.create_workflow(_scripted_patcher(proposals))
    )

    assert texts[0].startswith("Baseline: 5 Bandit findings, 8 failing tests")
    assert "rejected" in texts[1]
    assert sum("accepted" in t for t in texts[2:6]) == 4
    report_text = texts[-1]
    assert "| Bandit findings | 5 | 0 |" in report_text
    assert "| Tests passing | 9/17 | 17/17 |" in report_text
    assert "Patch attempts: 5 of 8." in report_text
    assert "```diff" in report_text
    assert "Nothing was committed" in report_text
    report = agent.build_report(final_state)
    assert report["files_changed"] == sorted(ALL_FIXES)
    assert len(report["exploit_tests_fixed"]) == 8
    assert (
        '+    query = "SELECT * FROM orders WHERE customer = ?"'
        in report["diff"]
    )
    assert Path(report["patch_file"]).read_text() == report["diff"]
    # The bundled target is never modified in place.
    assert _tree_hash(workspace.BUNDLED_TARGET) == before_hash


def test_workflow_stops_when_the_budget_is_spent(
    state: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_PATCH_ATTEMPTS", "2")
    proposals = [("orders/db.py", DB_BROKEN)] * 2
    texts, final_state = _run(
        agent.create_workflow(_scripted_patcher(proposals))
    )
    assert sum(t.startswith("Attempt") and "rejected" in t for t in texts) == 2
    assert "Patch attempts: 2 of 2." in texts[-1]
    assert "# no accepted changes" in texts[-1]
    assert len(final_state["cm_attempts"]) == 2


def test_root_agent_is_the_gemini_workflow() -> None:
    assert isinstance(agent.root_agent, Workflow)
    assert isinstance(agent.app, App)
    assert agent.app.root_agent is agent.root_agent
    patch_agent = agent.create_patch_agent()
    assert patch_agent.mode == "single_turn"
    assert patch_agent.output_schema is agent.PatchProposal
