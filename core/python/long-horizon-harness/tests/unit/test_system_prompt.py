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

"""System prompt 3-tier assembly — deterministic tests.

Covers ``build_system_prompt_parts`` (the 3-tier assembly) and the
context-file AGENTS.md / CLAUDE.md / .cursorrules loaders.

The 3 tiers (*"Agent instruction (stable) +
before_model_callback (context + volatile)"*):

  * stable   — ``Agent.instruction`` (already set at horizon/agent.py:76).
               Never rewritten by the callback. Prefix-cache friendly.
  * context  — discovered AGENTS.md / CLAUDE.md / .cursorrules under
               cwd. Session-stable.
  * volatile — iteration counter + last error +
               **date-only** timestamp. Re-rendered every turn.

The date-only convention is the load-bearing prefix-cache trick:
date-only so the system prompt is byte-stable for the full day.
Minute-precision changes invalidate prefix-cache KV on every rebuild path.

These tests assert on STATE (the bytes of
``llm_request.config.system_instruction`` after the callback runs),
never on LLM response content. Pytest is for code correctness; LLM
behavior validation belongs in evalsets.

**Implementation surface** at
``horizon/conversation/system_prompt.py``:

    def discover_context_files(cwd: str | Path) -> list[tuple[str, str]]:
        '''Return [(filename, content), ...] for AGENTS.md / CLAUDE.md /
        .cursorrules found at the top level of cwd. Empty list if none.'''

    def build_context_block(cwd: str | Path) -> str | None:
        '''Compose the context tier. None if no context files exist.'''

    async def system_prompt_assembly_callback(
        callback_context, llm_request
    ) -> None:
        '''before_model_callback that assembles context + volatile and
        appends them to llm_request.config.system_instruction. Returns
        None (pass-through). Mutates llm_request in place.'''

**Deliberately out of scope for this sample:**
GOOGLE_MODEL_OPERATIONAL_GUIDANCE, OPENAI_MODEL_EXECUTION_GUIDANCE,
WSL/Termux/platform hints, SOUL.md loader, kanban tool guidance,
mixture-of-agents, Alibaba model-name workaround.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.adk.models import LlmRequest

from horizon.conversation.soul_loader import (
    DEFAULT_AGENT_IDENTITY,
    load_soul_identity,
)
from horizon.conversation.system_prompt import (
    ARTIFACT_HTML_GUIDANCE,
    FAILURE_LOOP_GUIDANCE,
    FILESYSTEM_WRITE_GUIDANCE,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    MEMORY_GUIDANCE,
    OUTPUT_STYLE_GUIDANCE,
    PLANNING_DISCIPLINE_GUIDANCE,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
    SUBAGENT_USAGE_GUIDANCE,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    TOOL_USE_SAFETY_GUIDANCE,
    WEB_RESEARCH_CITATION_GUIDANCE,
    build_stable_tier,
    discover_context_files,
    system_prompt_assembly_callback,
)

pytestmark = pytest.mark.asyncio


STABLE_INSTRUCTION = (
    "You are a helpful AI assistant. You remember user preferences across "
    "sessions via add_memory."
)


def _seed_request(stable: str = STABLE_INSTRUCTION) -> LlmRequest:
    """LlmRequest with the stable tier already on config.system_instruction.

    Mirrors what ADK does in production: Agent.instruction is materialized
    onto llm_request.config.system_instruction before any
    before_model_callback runs.
    """
    request = LlmRequest(model="gemini-3.6-flash")
    request.config.system_instruction = stable
    return request


def _fake_callback_context(
    *, cwd: str | Path | None = None, state: dict | None = None
) -> object:
    """Duck-typed CallbackContext exposing .state and (optionally) a cwd hint.

    The production callback resolves cwd from os.getcwd() unless overridden;
    tests pin to a tmp_path via the ``cwd`` attribute the production code
    is expected to consult (or via os.getcwd patching at the test level).
    """
    return SimpleNamespace(state=state if state is not None else {}, cwd=cwd)


# =============================================================================
# 1. Ordering — stable, then context, joined by \n\n (volatile rides the tail)
# =============================================================================


async def test_two_tier_ordering_stable_then_context(tmp_path: Path):
    """Assembled system_instruction is stable + '\\n\\n' + context.
    Volatile no longer rides the instruction (it ships via the
    <system-reminder> tail), so the date line must be absent here.
    Pinned with a real context file on disk so this exercises
    discover_context_files."""
    (tmp_path / "AGENTS.md").write_text(
        "## Project rule: tests must hit InMemoryMemoryService.\n"
    )

    request = _seed_request()
    ctx = _fake_callback_context(cwd=tmp_path, state={"iteration": 0})

    with patch("os.getcwd", return_value=str(tmp_path)):
        result = await system_prompt_assembly_callback(ctx, request)

    assert result is None
    assembled = request.config.system_instruction
    assert isinstance(assembled, str)

    stable_idx = assembled.index(STABLE_INSTRUCTION)
    context_idx = assembled.index("InMemoryMemoryService")
    assert stable_idx < context_idx, (
        f"Tier ordering broken — got stable@{stable_idx}, "
        f"context@{context_idx} in:\n{assembled}"
    )
    # Volatile date line must NOT be in the instruction anymore.
    date_marker = datetime.now().strftime("%A, %B %d, %Y")
    assert date_marker not in assembled


# =============================================================================
# 2. Instruction stays byte-identical when volatile state changes mid-session
# =============================================================================


async def test_instruction_stable_when_iteration_changes(tmp_path: Path):
    """Calling the callback twice in the same session with iteration mutated
    between calls produces a byte-identical system_instruction — the volatile
    tier moved to the <system-reminder> tail, so iteration changes must NOT
    perturb the cached prefix."""
    state = {"iteration": 0}
    ctx = _fake_callback_context(cwd=tmp_path, state=state)

    request_turn_1 = _seed_request()
    with patch("os.getcwd", return_value=str(tmp_path)):
        await system_prompt_assembly_callback(ctx, request_turn_1)
    prompt_turn_1 = request_turn_1.config.system_instruction

    state["iteration"] = 3

    request_turn_2 = _seed_request()
    with patch("os.getcwd", return_value=str(tmp_path)):
        await system_prompt_assembly_callback(ctx, request_turn_2)
    prompt_turn_2 = request_turn_2.config.system_instruction

    assert prompt_turn_1 == prompt_turn_2, (
        "Prefix-cache invariant broken — iteration change leaked into the "
        "system_instruction. Volatile must ride the reminder tail."
    )
    assert "Iteration: 3" not in prompt_turn_2


# =============================================================================
# 3. Prefix-cache invariant — byte-identical when nothing changes
# =============================================================================


async def test_byte_identical_when_state_unchanged_within_day(tmp_path: Path):
    """Two calls in the same calendar day with NO state mutation produce
    byte-identical system_instruction. This is the prefix-cache invariant —
    if it breaks, every turn invalidates the model's KV cache.

    Date-only timestamp is the load-bearing detail here. If volatile
    included minute-precision time, this test would fail."""
    (tmp_path / "AGENTS.md").write_text("Project rules.\n")
    state = {"iteration": 0}
    ctx = _fake_callback_context(cwd=tmp_path, state=state)

    request_a = _seed_request()
    request_b = _seed_request()

    with patch("os.getcwd", return_value=str(tmp_path)):
        await system_prompt_assembly_callback(ctx, request_a)
        await system_prompt_assembly_callback(ctx, request_b)

    assert (
        request_a.config.system_instruction
        == request_b.config.system_instruction
    ), (
        "Prefix-cache invariant broken — two calls in the same day with "
        "identical state produced different system_instruction. Likely "
        "cause: volatile tier embeds minute-precision time, hash, or "
        "session-id instead of date-only."
    )


# =============================================================================
# 4. Missing context-tier files — section omitted, no boilerplate
# =============================================================================


async def test_missing_context_files_omits_context_tier_cleanly(
    tmp_path: Path,
):
    """When cwd has no AGENTS.md / CLAUDE.md / .cursorrules, the context
    tier is omitted — no 'Context: (none)' placeholder, no triple
    newlines, no empty section header. The assembled prompt is just the
    stable tier (volatile rides the reminder tail, not the instruction)."""
    # tmp_path is empty — no context files.
    assert discover_context_files(tmp_path) == [], (
        "Sanity: empty tmp_path must yield no context files."
    )

    request = _seed_request()
    ctx = _fake_callback_context(cwd=tmp_path, state={"iteration": 0})
    with patch("os.getcwd", return_value=str(tmp_path)):
        await system_prompt_assembly_callback(ctx, request)

    assembled = request.config.system_instruction

    # Pin the negative: no boilerplate strings.
    assert "(none)" not in assembled
    assert "No context files" not in assembled.lower()
    assert "Context: " not in assembled  # no labeled-but-empty section
    # Pin the negative: no triple-newline gap (would happen if an empty
    # context block were joined by "\n\n" on both sides).
    assert "\n\n\n" not in assembled, (
        f"Triple-newline gap detected — context tier rendered an empty "
        f"block. Got:\n{assembled!r}"
    )


# =============================================================================
# 5. Volatile content — iteration, last error, date-only timestamp
# =============================================================================


async def test_system_prompt_callback_does_not_append_volatile_to_instruction():
    """Volatile state (iteration/last_error/date) must NOT land in
    system_instruction — it now rides the <system-reminder> tail.

    The cached prefix must be byte-identical between a turn at iteration 1
    and a turn at iteration 9 with a last_error set.
    """
    ctx_early = _fake_callback_context(state={"iteration": 1})
    req_early = _seed_request()
    await system_prompt_assembly_callback(ctx_early, req_early)

    ctx_late = _fake_callback_context(
        state={"iteration": 9, "last_error": "terminal blew up"}
    )
    req_late = _seed_request()
    await system_prompt_assembly_callback(ctx_late, req_late)

    instr_early = req_early.config.system_instruction or ""
    instr_late = req_late.config.system_instruction or ""

    # No volatile leakage into the cached prefix.
    assert "Iteration: 9" not in instr_late
    assert "terminal blew up" not in instr_late
    # And the today-date line is gone from the instruction.
    assert datetime.now().strftime("%A, %B %d, %Y") not in instr_late
    # The stable prefix is identical across the two turns despite mutated state.
    assert instr_early == instr_late


# =============================================================================
# 7. SOUL.md identity loader
# =============================================================================


async def test_load_soul_identity_uses_soul_md_when_present(tmp_path: Path):
    soul = tmp_path / "SOUL.md"
    soul.write_text("You are Project lha, a careful and concise assistant.\n")

    result = load_soul_identity(soul_path=soul)

    assert result == "You are Project lha, a careful and concise assistant."
    assert result != DEFAULT_AGENT_IDENTITY


async def test_load_soul_identity_falls_back_when_missing(tmp_path: Path):
    missing = tmp_path / "does-not-exist.md"

    result = load_soul_identity(soul_path=missing)

    assert result == DEFAULT_AGENT_IDENTITY


async def test_load_soul_identity_falls_back_when_empty(tmp_path: Path):
    empty = tmp_path / "SOUL.md"
    empty.write_text("   \n\n")

    result = load_soul_identity(soul_path=empty)

    assert result == DEFAULT_AGENT_IDENTITY


# =============================================================================
# 8. Tool-conditional guidance blocks
# =============================================================================


async def test_memory_guidance_present_when_memory_tool_loaded(tmp_path: Path):
    stable = build_stable_tier(
        tool_names=["add_memory", "preload_memory"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert MEMORY_GUIDANCE in stable


async def test_memory_guidance_absent_when_memory_tool_not_loaded(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file", "terminal"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert MEMORY_GUIDANCE not in stable


async def test_skills_guidance_present_when_skill_tool_loaded(tmp_path: Path):
    stable = build_stable_tier(
        tool_names=["skill"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert SKILLS_GUIDANCE in stable


async def test_skills_guidance_absent_when_skill_tool_not_loaded(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file", "terminal"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert SKILLS_GUIDANCE not in stable


async def test_artifact_html_guidance_present_when_tool_loaded(tmp_path: Path):
    stable = build_stable_tier(
        tool_names=["artifact"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert ARTIFACT_HTML_GUIDANCE in stable


async def test_artifact_html_guidance_absent_when_tool_not_loaded(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file", "terminal"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert ARTIFACT_HTML_GUIDANCE not in stable


# =============================================================================
# 9. Google model operational guidance
# =============================================================================


async def test_google_operational_guidance_present_for_gemini_model(
    tmp_path: Path,
):
    # Tools must be present for enforcement (and its google sibling) to inject;
    # this mirrors the agent's valid_tool_names gate.
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert GOOGLE_MODEL_OPERATIONAL_GUIDANCE in stable


async def test_google_operational_guidance_absent_for_non_google_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert GOOGLE_MODEL_OPERATIONAL_GUIDANCE not in stable


async def test_google_operational_guidance_absent_when_no_tools_loaded(
    tmp_path: Path,
):
    """No tools → no enforcement, no google operational guidance.

    This guards the conversational default: a fresh agent answering 'hi' with
    no tools attached should not get a system prompt full of 'use your tools'
    directives.
    """
    stable = build_stable_tier(
        tool_names=[],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert GOOGLE_MODEL_OPERATIONAL_GUIDANCE not in stable
    assert TOOL_USE_ENFORCEMENT_GUIDANCE not in stable


async def test_tool_use_enforcement_present_for_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert TOOL_USE_ENFORCEMENT_GUIDANCE in stable


async def test_tool_use_enforcement_absent_for_non_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert TOOL_USE_ENFORCEMENT_GUIDANCE not in stable


async def test_tool_use_enforcement_env_force_off(monkeypatch, tmp_path: Path):
    """LHA_TOOL_USE_ENFORCEMENT=off suppresses enforcement even on Gemini."""
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "off")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert TOOL_USE_ENFORCEMENT_GUIDANCE not in stable
    assert GOOGLE_MODEL_OPERATIONAL_GUIDANCE not in stable


async def test_tool_use_enforcement_env_force_on(monkeypatch, tmp_path: Path):
    """LHA_TOOL_USE_ENFORCEMENT=on injects enforcement even on Claude."""
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "on")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert TOOL_USE_ENFORCEMENT_GUIDANCE in stable
    # Google block stays gated on model family, not the env override.
    assert GOOGLE_MODEL_OPERATIONAL_GUIDANCE not in stable


# =============================================================================
# Output style + tool-use safety — universal stable-tier additions
# (always fire with TOOL_USE_ENFORCEMENT_GUIDANCE, no model gating)
# =============================================================================


async def test_output_style_guidance_present_for_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert OUTPUT_STYLE_GUIDANCE in stable


async def test_output_style_guidance_absent_for_non_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert OUTPUT_STYLE_GUIDANCE not in stable


async def test_output_style_guidance_env_force_off(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "off")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert OUTPUT_STYLE_GUIDANCE not in stable


async def test_output_style_guidance_env_force_on(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "on")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert OUTPUT_STYLE_GUIDANCE in stable


async def test_tool_use_safety_guidance_present_for_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert TOOL_USE_SAFETY_GUIDANCE in stable


async def test_tool_use_safety_guidance_absent_for_non_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert TOOL_USE_SAFETY_GUIDANCE not in stable


async def test_tool_use_safety_guidance_env_force_off(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "off")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert TOOL_USE_SAFETY_GUIDANCE not in stable


async def test_tool_use_safety_guidance_env_force_on(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "on")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert TOOL_USE_SAFETY_GUIDANCE in stable


async def test_subagent_usage_guidance_present_for_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert SUBAGENT_USAGE_GUIDANCE in stable


async def test_subagent_usage_guidance_absent_for_non_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert SUBAGENT_USAGE_GUIDANCE not in stable


async def test_subagent_usage_guidance_env_force_off(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "off")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert SUBAGENT_USAGE_GUIDANCE not in stable


async def test_subagent_usage_guidance_env_force_on(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "on")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert SUBAGENT_USAGE_GUIDANCE in stable


async def test_planning_discipline_guidance_present_for_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert PLANNING_DISCIPLINE_GUIDANCE in stable


async def test_planning_discipline_guidance_absent_for_non_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert PLANNING_DISCIPLINE_GUIDANCE not in stable


async def test_planning_discipline_guidance_env_force_off(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "off")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert PLANNING_DISCIPLINE_GUIDANCE not in stable


async def test_planning_discipline_guidance_env_force_on(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "on")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert PLANNING_DISCIPLINE_GUIDANCE in stable


async def test_failure_loop_guidance_present_for_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert FAILURE_LOOP_GUIDANCE in stable


async def test_failure_loop_guidance_absent_for_non_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert FAILURE_LOOP_GUIDANCE not in stable


async def test_failure_loop_guidance_env_force_off(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "off")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert FAILURE_LOOP_GUIDANCE not in stable


async def test_failure_loop_guidance_env_force_on(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "on")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert FAILURE_LOOP_GUIDANCE in stable


async def test_filesystem_write_guidance_present_for_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert FILESYSTEM_WRITE_GUIDANCE in stable


async def test_filesystem_write_guidance_absent_for_non_enforcement_model(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert FILESYSTEM_WRITE_GUIDANCE not in stable


async def test_filesystem_write_guidance_env_force_off(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "off")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert FILESYSTEM_WRITE_GUIDANCE not in stable


async def test_filesystem_write_guidance_env_force_on(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("LHA_TOOL_USE_ENFORCEMENT", "on")

    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="claude-3-5-sonnet",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert FILESYSTEM_WRITE_GUIDANCE in stable


async def test_memory_guidance_lists_do_save_categories():
    """Memory-write discipline: positive examples should appear so the model
    knows what kinds of facts are worth persisting."""
    assert "durable user preferences" in MEMORY_GUIDANCE
    assert "stated rationale" in MEMORY_GUIDANCE


async def test_memory_guidance_warns_against_derivable_facts():
    """Memory-write discipline: negative examples should appear so the model
    does not snapshot info already in git history or the codebase."""
    assert "derivable from `git log`" in MEMORY_GUIDANCE
    assert "in-flight TODOs" in MEMORY_GUIDANCE


async def test_memory_guidance_requires_why_clause():
    """Memory-write discipline: the format rule should require a 'why' clause
    so memories remain interpretable when context shifts."""
    assert "plus a short 'why'" in MEMORY_GUIDANCE


async def test_root_agent_instruction_notes_user_profile_is_offline_curated():
    """The agent instruction should tell the live agent that the
    `## User Profile` block is a read-only rollup maintained offline by
    dream-review — so it doesn't try to update it itself or treat its
    absence as a gap. scope='user' atomic captures still happen here."""
    from horizon.conversation.system_prompt import ROOT_AGENT_INSTRUCTION

    assert "curated snapshot of who the user is" in ROOT_AGENT_INSTRUCTION


async def test_web_research_citation_guidance_present_when_tool_loaded(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["web_research"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert WEB_RESEARCH_CITATION_GUIDANCE in stable


async def test_web_research_citation_guidance_absent_when_tool_not_loaded(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert WEB_RESEARCH_CITATION_GUIDANCE not in stable


async def test_google_operational_guidance_no_longer_contains_conciseness():
    """Conciseness moved to OUTPUT_STYLE_GUIDANCE (universal). Catches an
    accidental re-add to the Google-only block, which would double-issue
    the rule on Gemini and leave non-Google models without it."""
    lowered = GOOGLE_MODEL_OPERATIONAL_GUIDANCE.lower()
    assert "concise" not in lowered
    assert "brevity" not in lowered


async def test_session_search_guidance_present_when_tool_loaded(tmp_path: Path):
    stable = build_stable_tier(
        tool_names=["session_search"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert SESSION_SEARCH_GUIDANCE in stable


async def test_session_search_guidance_absent_when_tool_not_loaded(
    tmp_path: Path,
):
    stable = build_stable_tier(
        tool_names=["read_file"],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert SESSION_SEARCH_GUIDANCE not in stable


# =============================================================================
# 12. Context discovery is first-match-wins, wrapped with header
# =============================================================================


async def test_context_discovery_is_first_match_wins(tmp_path: Path):
    """When multiple context files are present, only the highest-priority one
    loads. Loading all three concurrently bloats the prefix and gives the
    model conflicting directives.
    """
    (tmp_path / "AGENTS.md").write_text("AGENTS rule: be careful.\n")
    (tmp_path / "CLAUDE.md").write_text("CLAUDE rule: be bold.\n")
    (tmp_path / ".cursorrules").write_text("cursor rule: be terse.\n")

    found = discover_context_files(tmp_path)

    assert len(found) == 1
    assert found[0][0] == "AGENTS.md"


async def test_context_discovery_prefers_lha_md_over_agents_md(tmp_path: Path):
    (tmp_path / ".horizon.md").write_text("lha config wins.\n")
    (tmp_path / "AGENTS.md").write_text("AGENTS rule: be careful.\n")

    found = discover_context_files(tmp_path)

    assert len(found) == 1
    assert found[0][0] == ".horizon.md"


async def test_context_block_has_project_context_header(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("project rules here\n")
    from horizon.conversation.system_prompt import build_context_block

    block = build_context_block(tmp_path)

    assert block is not None
    assert block.startswith("# Project Context")
    assert "## AGENTS.md" in block
    assert "project rules here" in block


# =============================================================================
# 10. Environment hints
# =============================================================================


async def test_environment_hint_detects_python_project(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")

    stable = build_stable_tier(
        tool_names=[],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert "Python project" in stable
    assert str(tmp_path) in stable


async def test_environment_hint_includes_user_and_home(
    tmp_path: Path, monkeypatch
):
    """User + home are needed so the model can construct absolute paths
    (~/foo) without an exploratory `whoami`/`echo $HOME` terminal call."""
    from horizon.conversation.system_prompt import _clear_cli_probe_cache

    monkeypatch.setenv("LHA_ENVIRONMENT_BACKEND", "local")
    _clear_cli_probe_cache()
    stable = build_stable_tier(
        tool_names=[],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    import getpass

    assert f"User: {getpass.getuser()}" in stable
    assert f"Home: {Path.home()}" in stable


async def test_environment_hint_includes_cli_versions(
    tmp_path: Path, monkeypatch
):
    """python3 is always present in this repo's dev env; its version line
    must appear so the model can pick the right invocation without probing."""
    from horizon.conversation.system_prompt import _clear_cli_probe_cache

    monkeypatch.setenv("LHA_ENVIRONMENT_BACKEND", "local")
    _clear_cli_probe_cache()
    stable = build_stable_tier(
        tool_names=[],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    # Per-CLI lines start with `- <name>:` so any subset that resolves shows up.
    assert "Available CLIs:" in stable
    assert "- python3:" in stable


async def test_environment_hint_omits_host_details_in_sandbox_mode(
    tmp_path: Path, monkeypatch
):
    """In sandbox mode the agent runs in a Linux container, so host
    user/home/macOS/CLI probes describe the wrong machine — they must
    not leak into the prompt."""
    from horizon.conversation.system_prompt import _clear_cli_probe_cache

    monkeypatch.setenv("LHA_ENVIRONMENT_BACKEND", "sandbox")
    _clear_cli_probe_cache()
    stable = build_stable_tier(
        tool_names=[],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    import getpass

    assert "Sandbox: Linux container." in stable
    assert str(tmp_path) in stable
    assert f"User: {getpass.getuser()}" not in stable
    assert f"Home: {Path.home()}" not in stable
    assert "Available CLIs:" not in stable
    assert "macOS" not in stable


async def test_environment_hint_omits_missing_clis(tmp_path: Path, monkeypatch):
    """A CLI that fails to run (e.g. shutil.which returns None) must not
    appear in the hint and must not raise."""
    import shutil

    from horizon.conversation.system_prompt import _clear_cli_probe_cache

    real_which = shutil.which

    def fake_which(name: str, *args, **kwargs):
        if name == "uv":
            return None
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", fake_which)
    _clear_cli_probe_cache()

    stable = build_stable_tier(
        tool_names=[],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )

    assert "- uv:" not in stable


async def test_environment_hint_cli_probe_is_cached(
    tmp_path: Path, monkeypatch
):
    """The CLI probe runs subprocesses — it MUST cache per-process so two
    stable-tier builds don't reshell `--version` calls every time."""
    import subprocess

    from horizon.conversation import system_prompt as sp

    sp._clear_cli_probe_cache()
    call_count = {"n": 0}
    real_run = subprocess.run

    def counting_run(*args, **kwargs):
        call_count["n"] += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", counting_run)

    sp.build_stable_tier(
        tool_names=[],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )
    first_count = call_count["n"]

    sp.build_stable_tier(
        tool_names=[],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )
    assert call_count["n"] == first_count, (
        "CLI probe must be cached — repeated stable-tier builds must not "
        "reshell subprocess.run for each CLI"
    )


async def test_environment_hint_byte_stable_across_calls(tmp_path: Path):
    """Two back-to-back build_stable_tier calls with identical inputs must
    return byte-identical strings — required for prefix-cache warmth."""
    from horizon.conversation.system_prompt import _clear_cli_probe_cache

    _clear_cli_probe_cache()
    first = build_stable_tier(
        tool_names=[],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )
    second = build_stable_tier(
        tool_names=[],
        model_name="gemini-3.6-flash",
        cwd=tmp_path,
        soul_path=tmp_path / "no-soul.md",
    )
    assert first == second


# =============================================================================
# 11. Stable tier is byte-stable across two callback invocations (cached)
# =============================================================================


async def test_stable_tier_cached_in_state_across_calls(tmp_path: Path):
    """When the callback runs twice on the same session.state, the stable
    tier MUST be byte-identical AND the cache key must be populated after
    turn 1. Prefix-cache invariant."""
    from horizon.conversation.system_prompt import _STABLE_TIER_STATE_KEY

    state: dict = {"iteration": 0}
    ctx = SimpleNamespace(state=state, cwd=tmp_path)

    request_a = LlmRequest(model="gemini-3.6-flash")
    request_b = LlmRequest(model="gemini-3.6-flash")

    with patch("os.getcwd", return_value=str(tmp_path)):
        await system_prompt_assembly_callback(ctx, request_a)
        cached_after_first = state.get(_STABLE_TIER_STATE_KEY)
        await system_prompt_assembly_callback(ctx, request_b)

    assert cached_after_first, (
        "Stable tier must be cached in state after turn 1."
    )
    assert (
        request_a.config.system_instruction
        == request_b.config.system_instruction
    )
    # Stable tier substring must appear in both — identity is the first block.
    assert DEFAULT_AGENT_IDENTITY in request_a.config.system_instruction
    assert DEFAULT_AGENT_IDENTITY in request_b.config.system_instruction


async def test_stable_tier_assembled_when_instruction_empty(tmp_path: Path):
    """Production case: Agent.instruction is unset (lead removes the
    hardcoded string), so llm_request.config.system_instruction starts
    empty. The callback must assemble + install the stable tier."""
    state: dict = {"iteration": 0}
    ctx = SimpleNamespace(state=state, cwd=tmp_path)

    request = LlmRequest(model="gemini-3.6-flash")

    with patch("os.getcwd", return_value=str(tmp_path)):
        await system_prompt_assembly_callback(ctx, request)

    assembled = request.config.system_instruction
    assert isinstance(assembled, str) and assembled.strip()
    assert DEFAULT_AGENT_IDENTITY in assembled
