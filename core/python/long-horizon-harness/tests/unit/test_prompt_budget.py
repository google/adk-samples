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

"""Ratcheting size budgets for horizon's per-turn fixed prompt prefix.

Values start at today's measured baseline (2026-08-12) and only ever go
down as later tasks in the prompt-minimalism plan shrink each component.
This test must import the exact helpers `.data/minimalism/measure_prefix.py`
uses, so the two can never quietly disagree about what "the prefix" means.

No trajectory grader backs this work (see Task 1 of the plan for why: only
26 of 98 evalset invocations carry a real tool-use trajectory, 11 of those
with empty args, so a bare threshold would score near-zero noise). These
ratchets, plus the registry consistency tests, are the size and tool-choice
safety net instead.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
from google.adk.models import LlmRequest

from horizon.agent import _SKILL_TOOLSET, root_agent
from horizon.conversation.system_prompt import build_static_instruction
from horizon.tools import names

# Measured 2026-08-12 via `uv run python .data/minimalism/measure_prefix.py`.
# Lower these only as the task that shrinks the corresponding component
# lands (Task 6/7 landed the instruction+tier move; Task 8 for the preamble,
# Task 9 for the index, Task 3/4/5/9/10 for schemas).
# Raised from 8_000/8_600 (final-review Fix 5): agent.py's tool_names list
# was silently dropping the skill toolset's tools (a BaseToolset has no
# .name/__name__ of its own), so SKILLS_GUIDANCE never actually shipped in
# the real deployed prompt — only in tests that named "load_skill" by hand.
# Fixing that wiring bug correctly raised the measured total by ~190 chars.
# The real budget is the 27,000-char total prefix (measured 25,654), not
# this per-tier number; do not shrink real guidance to hit a stale ratchet.
MAX_STATIC_INSTRUCTION_CHARS = (
    8_300  # Agent.static_instruction, no code executor
)
MAX_STATIC_INSTRUCTION_CHARS_WITH_EXECUTOR = 8_900
# Lowered from 2,100: Task 8 replaced ADK's ~2 KB tutorial preamble with a
# one-line pointer via HorizonSkillToolset (measured 115).
MAX_SKILLS_PREAMBLE_CHARS = 200
# Lowered from 3,250: Task 9 cut C capped every builtin skill's frontmatter
# description at 200 chars (measured rendered index: 1,333).
MAX_SKILLS_INDEX_CHARS = 1_400  # rendered, XML-escaped, not raw frontmatter
# Task 10 slimmed every over-budget tool description per the policy in
# horizon/tools/__init__.py, then ADK_DISABLE_JSON_SCHEMA_FOR_FUNC_DECL
# (agent.py) dropped 2,812 chars of pydantic schema artifacts. Measured
# 13,206; the design's target was 17,000.
MAX_TOTAL_SCHEMA_CHARS = 13_600  # static declarations + rendered dynamic suffix

# Per-tool description caps (Task 10). Dispatch tools carry multiple
# actions in one description and get more room; everything else is a
# single-purpose tool and must fit in the simple cap. memory joined this
# set when session_search folded into memory(action='search') — it now
# dispatches on action like process/subagent/artifact/routine do.
MAX_SIMPLE_DESC_CHARS = 400
MAX_DISPATCH_DESC_CHARS = 900
DISPATCH_TOOLS = frozenset(
    {names.PROCESS, names.SUBAGENT, names.ARTIFACT, names.ROUTINE, names.MEMORY}
)
# Lowered from 41,000: Task 8/9's skills-surface cuts (preamble, index,
# catalog splice, load_skill merge) dropped the measured total from 38,859
# to 32,077. Task 10/11/12 (docstrings + rename) took it to 24,459, but this
# constant was never re-ratcheted after that — 32,500 let the prefix regress
# 33% with a fully green suite (final-review Fix 6). Final-review Fix 5 then
# fixed a real wiring bug (agent.py silently dropped the skill toolset's
# tool names, so SKILLS_GUIDANCE never shipped) that correctly raised the
# measured total to 25,654. Margin above that, not a new ceiling for later
# growth; the design's actual budget is 27,000.
MAX_TOTAL_PREFIX_CHARS = 23_500

# Per-block caps from the Task 7 guidance consolidation (design v4, "How
# static_instruction reaches 8,000"). Checked directly against the
# constants, not the assembled instruction, so a regression names its block.
MAX_MEMORY_BLOCK_CHARS = 1_400
MAX_ACTING_BLOCK_CHARS = 1_200
MAX_SAFETY_BLOCK_CHARS = 1_100
MAX_STYLE_BLOCK_CHARS = 900


async def _measure() -> dict[str, int]:
    tools = await root_agent.canonical_tools()
    tool_names = [t.name for t in tools]

    static = 0
    for tool in tools:
        decl = tool._get_declaration()
        if decl is not None:
            static += len(decl.model_dump_json(exclude_none=True))

    req = LlmRequest()
    await _SKILL_TOOLSET.process_llm_request(tool_context=None, llm_request=req)
    skills_block = req.config.system_instruction or ""
    idx = skills_block.find("<available_skills>")
    preamble = idx if idx >= 0 else len(skills_block)
    index = len(skills_block) - preamble if idx >= 0 else 0

    try:
        from horizon.subagents.descriptions import _build_suffix

        suffix = len(_build_suffix())
    except Exception:
        suffix = 0
    subagent_tool_count = len(
        [n for n in tool_names if n in {"delegate", "agent", "subagent"}]
    )
    dynamic = suffix * subagent_tool_count

    # Read the assembled instruction off the built agent (has_code_executor=
    # False in this repo's default dev config), so this can never disagree
    # with what the app actually serves, and separately at True so a real
    # deployment with CODE_SANDBOX_RESOURCE_NAME set is measured too.
    static_instruction = len(root_agent.static_instruction or "")
    static_instruction_with_executor = len(
        build_static_instruction(
            tool_names=tool_names,
            model_name="gemini-3.6-flash",
            has_code_executor=True,
        )
    )

    total = static_instruction + preamble + index + static + dynamic
    return {
        "static_instruction": static_instruction,
        "static_instruction_with_executor": static_instruction_with_executor,
        "skills_preamble": preamble,
        "skills_index": index,
        "total_schema": static + dynamic,
        "total_prefix": total,
    }


@pytest.fixture(scope="module")
def measured() -> dict[str, int]:
    return asyncio.run(_measure())


def test_static_instruction_within_budget(measured):
    assert measured["static_instruction"] <= MAX_STATIC_INSTRUCTION_CHARS, (
        measured["static_instruction"]
    )


def test_static_instruction_within_budget_with_code_executor(measured):
    # CODE_EXECUTION_GUIDANCE only injects when has_code_executor=True;
    # measuring only the False case would let a real deployment with
    # CODE_SANDBOX_RESOURCE_NAME set exceed the design's own cap unobserved.
    assert (
        measured["static_instruction_with_executor"]
        <= MAX_STATIC_INSTRUCTION_CHARS_WITH_EXECUTOR
    ), measured["static_instruction_with_executor"]


def test_skills_preamble_within_budget(measured):
    assert measured["skills_preamble"] <= MAX_SKILLS_PREAMBLE_CHARS, measured[
        "skills_preamble"
    ]


def test_skills_index_within_budget(measured):
    assert measured["skills_index"] <= MAX_SKILLS_INDEX_CHARS, measured[
        "skills_index"
    ]


def test_total_schema_within_budget(measured):
    assert measured["total_schema"] <= MAX_TOTAL_SCHEMA_CHARS, measured[
        "total_schema"
    ]


async def _tool_descriptions() -> dict[str, str]:
    # Measure what the model receives, not the raw docstring: ADK's
    # non-pydantic path keeps source indentation and
    # normalize_tool_schemas_callback strips it per request.
    tools = await root_agent.canonical_tools()
    out: dict[str, str] = {}
    for tool in tools:
        decl = tool._get_declaration()
        if decl is not None:
            out[tool.name] = inspect.cleandoc(decl.description or "")
    return out


def test_each_tool_description_within_budget():
    descriptions = asyncio.run(_tool_descriptions())
    over = {}
    for name, desc in descriptions.items():
        cap = (
            MAX_DISPATCH_DESC_CHARS
            if name in DISPATCH_TOOLS
            else MAX_SIMPLE_DESC_CHARS
        )
        if len(desc) > cap:
            over[name] = (len(desc), cap)
    assert not over, over


def test_total_prefix_within_budget(measured):
    assert measured["total_prefix"] <= MAX_TOTAL_PREFIX_CHARS, measured[
        "total_prefix"
    ]


def test_guidance_block_budgets():
    from horizon.conversation.system_prompt import (
        ACTING_GUIDANCE,
        MEMORY_GUIDANCE,
        SAFETY_GUIDANCE,
        STYLE_GUIDANCE,
    )

    assert len(MEMORY_GUIDANCE) <= MAX_MEMORY_BLOCK_CHARS, len(MEMORY_GUIDANCE)
    assert len(ACTING_GUIDANCE) <= MAX_ACTING_BLOCK_CHARS, len(ACTING_GUIDANCE)
    assert len(SAFETY_GUIDANCE) <= MAX_SAFETY_BLOCK_CHARS, len(SAFETY_GUIDANCE)
    assert len(STYLE_GUIDANCE) <= MAX_STYLE_BLOCK_CHARS, len(STYLE_GUIDANCE)
