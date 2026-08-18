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

"""Tool-name registry safety net.

A ban-list regex scan over horizon/**/*.py was tried first and rejected: it
flags 131 lines in 42 files, almost none of them actual tool names (the
`process` write action's `"write"` string, shell interpreter literals,
scheduler job-type literals, slash-command registrations). Fixing those
false positives would bind unrelated code to tool names, which is the
coupling this registry exists to prevent.

Instead, three positive tests verify the real invariant: every fail-closed
tool-name set is a subset of the live registry (test 1), the registry
equals the live tool set (test 2), and no dead legacy token survives
outside a small, explicit exception list of permanent survivors — ADK
BaseEnvironment methods, deliberately-retained internal helpers, permanent
TOOL_NAME_ALIASES keys, the unrenamed add_memory_tool.py module filename,
an external MCP server's own same-named tool, and eval_id/rubric_id labels
(test 3, flipped to a real assertion once Task 11's rename and Task 12's
surface cleanup landed).

A later review pass (post tail-cuts items 1-4) tried extending test 3's
token list to "reload", "patch", "terminal", and "background=True" and hit
the scan's own bug: `web/**/*.ts(x)` globbed vendored `node_modules` too
(4,289 of 4,516 matches), which the exclusion below now fixes. Even with
that fixed, those four tokens still hit 40+ files each, dominated by
unrelated meanings (web page/HMR/config reload, unittest.mock.patch, the
web UI's own terminal/patch concepts, e2e test files) — the exact
false-positive class this module already rejected the regex scan for, so
they stay out. "background=True" is worse than noisy: `subagent(
  background=True)` is a real, current, valid parameter with the identical
literal spelling, so the token cannot distinguish a dead bash reference
from a live subagent one. "session_search", "old_string", "new_string",
"replace_all", and "as_media" passed the same measurement cleanly and were
added.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from horizon.tools import names

pytestmark = pytest.mark.asyncio

_HORIZON_ROOT = Path(__file__).resolve().parents[2] / "horizon"
_DEFAULT_POLICIES_JSONL = (
    _HORIZON_ROOT / "guardrails" / "default_policies.jsonl"
)


def _fail_closed_sets() -> dict[str, set[str]]:
    from horizon.a2a import executor as a2a_executor
    from horizon.context import artifact_url_redaction as aur
    from horizon.context import summarizer as summ
    from horizon.context import tool_output_pruning as top
    from horizon.conversation import system_prompt as sp
    from horizon.guardrails import permission_rules as pr
    from horizon.guardrails.permission_guard import (
        READ_ONLY_TOOLS,
        SELF_CONFIRMING_TOOLS,
        SUBAGENT_TOOLS,
    )
    from horizon.memory import flush_fork as ff
    from horizon.memory import review_fork as rf
    from horizon.memory import skill_telemetry as st
    from horizon.subagents import delegate_builder as db
    from horizon.subagents import descriptions as desc
    from horizon.subagents import profiles as prof

    sets: dict[str, set[str]] = {
        "READ_ONLY_TOOLS": set(READ_ONLY_TOOLS),
        "SELF_CONFIRMING_TOOLS": set(SELF_CONFIRMING_TOOLS),
        "SUBAGENT_TOOLS": set(SUBAGENT_TOOLS),
        "_SANDBOX_WRITE_TOOLS": set(pr._SANDBOX_WRITE_TOOLS),
        "_BENIGN_SIDE_EFFECT_TOOLS": set(pr._BENIGN_SIDE_EFFECT_TOOLS),
        "_TOOLS_REQUIRING_NARROWING": set(pr._TOOLS_REQUIRING_NARROWING),
        "_DEFAULT_RULE_DICTS": {
            d["toolName"]
            for d in pr._DEFAULT_RULE_DICTS
            if d["toolName"] != "*"
        },
        "_REVIEW_TOOL_NAMES": set(rf._REVIEW_TOOL_NAMES),
        "_SKILL_MUTATE_TOOL_NAMES": set(rf._SKILL_MUTATE_TOOL_NAMES),
        "_FLUSH_TOOL_NAMES": set(ff._FLUSH_TOOL_NAMES),
        "_VIEW_TOOLS": set(st._VIEW_TOOLS),
        "_MUTATE_TOOLS": set(st._MUTATE_TOOLS),
        "_BLOCKED_TOOL_NAMES": set(db._BLOCKED_TOOL_NAMES),
        "_SUBAGENT_TOOL_NAMES": set(desc._SUBAGENT_TOOL_NAMES),
        "_MEMORY_TOOL_NAMES": set(sp._MEMORY_TOOL_NAMES),
        # _RECALL_PAST_SESSIONS_TOOL_NAMES and _ARTIFACT_TOOL_NAMES were
        # removed in Task 6 (folded unconditionally into STYLE_GUIDANCE);
        # _WEB_RESEARCH_TOOL_NAMES was removed the same way.
        # _SESSION_SEARCH_TOOL_NAMES is gone entirely: session_search folded
        # into memory(action='search'), and its guidance now lives inside
        # MEMORY_GUIDANCE, gated on _MEMORY_TOOL_NAMES above — no separate
        # gate to track. _SKILL_TOOL_NAMES is no longer a dead gate — it
        # points at a real registered tool (see test_system_prompt_gating.py).
        "_SKILL_TOOL_NAMES": set(sp._SKILL_TOOL_NAMES),
        # Cumulative file tracking across compactions (6c).
        "_READ_FILE_TOOLS": set(summ._READ_FILE_TOOLS),
        "_WRITE_FILE_TOOLS": set(summ._WRITE_FILE_TOOLS),
        # Security control: on a missed rename this silently starts letting
        # the model read credentialed 7-day V4 signed URLs (verification
        # review blocker 3).
        "artifact_url_redaction.ARTIFACT": {aur.ARTIFACT},
        # "adk_request_confirmation" is an ADK framework literal, not a
        # horizon tool name, so it's excluded the same way the "*" wildcard
        # and "skill" substring are excluded elsewhere in this function.
        "_FRAMEWORK_TOOL_NAMES": set(a2a_executor._FRAMEWORK_TOOL_NAMES)
        - {"adk_request_confirmation"},
        # "skill" is a substring match (names.LOAD_SKILL), not a tool name
        # itself, so it can't round-trip through names.ALL — exclude it here
        # the same way default_policies.jsonl excludes the "*" wildcard.
        "PROTECTED_TOOL_SUBSTRINGS": set(top.PROTECTED_TOOL_SUBSTRINGS)
        - {"skill"},
    }
    for profile_name, profile in prof.PROFILES.items():
        allowed = profile.allowed_tool_names
        if allowed:
            sets[f"PROFILES[{profile_name}].allowed_tool_names"] = set(allowed)
    # default_policies.jsonl is data, not Python, so its canonical_tool_name
    # values can't reference names.py's constants directly — policies.py's
    # loader aliases them at read time (_alias_canonical_tool_name), by
    # design, so the file itself never needs to change on a rename. Compare
    # the aliased form, matching what production actually resolves.
    sets["default_policies.jsonl"] = {
        names.apply_tool_aliases(json.loads(line)["canonical_tool_name"])
        for line in _DEFAULT_POLICIES_JSONL.read_text().splitlines()
        if line.strip()
    }
    return sets


async def test_no_fail_closed_set_references_a_missing_tool():
    # Task 6 fixed both prompt gates that used to name a dead tool
    # ("session_search", "skill"), so every set is now held to the same
    # standard — no more _KNOWN_DEAD_GATES carve-out.
    stray = {
        label: sorted(members - names.ALL)
        for label, members in _fail_closed_sets().items()
        if members - names.ALL
    }
    assert not stray, stray


async def test_registry_matches_registered_tools():
    from horizon.agent import root_agent

    live = {t.name for t in await root_agent.canonical_tools()}
    assert live == set(names.ALL), {
        "declared_not_registered": sorted(names.ALL - live),
        "registered_not_declared": sorted(live - names.ALL),
    }


# Unambiguous dead names only. "terminal", "patch", and "write_file" are
# excluded: they have legitimate non-tool meanings (a job-type literal, the
# unittest.mock function, a plain English word) that would make this scan
# noisy without the registry's help. "delegate" is deliberately excluded
# too, for the same reason, not an oversight: the internal delegate()
# callable behind the merged `subagent` tool is real and still named that
# (horizon/subagents/subagent.py's own docstring says so), so every current
# doc/architecture mention of "delegate" (docs/architecture.md,
# docs/extending.md, docs/security-model.md, README.md's "dynamic delegate +
# HITL resurfacing" interface name) describes that internal mechanism
# correctly, not a stale claim that `delegate` is a directly model-callable
# tool. Adding it as a scanned token would need nearly as many exceptions as
# real hits. The one genuinely stale claim found in this class (README.md
# describing "blocking delegate and fire-and-forget agent" as two separate
# model-facing tools) was fixed directly instead.
# "reload", "patch", and "terminal" were measured and left out (final review
# HIGH item): even with node_modules excluded, each still hits 40+ files
# dominated by unrelated meanings (web page/HMR/config reload, unittest.mock
# .patch plus the web PatchView component plus the English word, the
# terminal_exec.py module plus the web UI's terminal concept plus e2e test
# files) — exactly the false-positive class the module docstring already
# describes for "terminal"/"patch", now confirmed for "reload" too. Also left
# out: "background=True", because it is not just noisy but genuinely
# ambiguous — `subagent(background=True)` is a real, current, valid
# parameter with the identical literal spelling, so the token can't tell a
# dead bash reference from a live subagent one.
_LEGACY_TOKENS = (
    "read_file",
    "view_file",
    "add_memory",
    "recall_past_sessions",
    "write_todos",
    "repo_overview",
    "set_workspace_window",
    "report_to_maintainers",
    "load_skill_resource",
    "run_skill_script",
    "session_search",
    "old_string",
    "new_string",
    "replace_all",
    "as_media",
)

_LEGACY_SCAN_GLOBS = (
    "horizon/**/*.py",
    "horizon/builtin_skills/**/*.md",
    "tests/eval/evalsets/*.json",
    "web/**/*.ts",
    "web/**/*.tsx",
    "docs/**/*.md",
    # Root-level docs live outside docs/** and were missed until final
    # review Fix 8 caught two stale references (README.md's "delegate and
    # agent" tool pair, AGENTS.md's deleted self-report skill) that this
    # scan should have flagged but couldn't reach.
    "README.md",
    "AGENTS.md",
)

# Permanent, legitimate survivors after Task 11/12's cleanup — each is one
# of: (a) an ADK BaseEnvironment method (Environment.read_file), never
# renamed; (b) a deliberately-retained internal helper (read_file in
# file_ops.py, recall_past_sessions_entries), not the registered tool; (c) a
# permanent TOOL_NAME_ALIASES key in names.py, needed for old persisted
# permission rules; (d) the unrenamed add_memory_tool.py module filename; (e)
# an external MCP server's own same-named tool, unrelated to ours; (f) an
# eval_id/rubric_id label (not graded content — the graded text_property
# fields are clean); (g) historically-accurate prose about a deleted tool.
_LEGACY_TOKEN_EXCEPTIONS: dict[str, frozenset[str]] = {
    "horizon/tools/names.py": frozenset(_LEGACY_TOKENS),
    "horizon/tools/file_ops.py": frozenset({"read_file", "replace_all"}),
    "horizon/tools/__init__.py": frozenset({"read_file"}),
    "horizon/tools/read.py": frozenset({"read_file", "as_media"}),
    "horizon/tools/artifacts.py": frozenset({"read_file"}),
    "horizon/tools/skill_loader.py": frozenset({"read_file"}),
    "horizon/tools/past_sessions.py": frozenset({"recall_past_sessions"}),
    "horizon/guardrails/_overlay.py": frozenset({"read_file"}),
    "horizon/guardrails/permission_rules.py": frozenset({"read_file"}),
    "horizon/environment/sandbox.py": frozenset({"read_file"}),
    "horizon/api/uploads.py": frozenset({"read_file"}),
    "horizon/sandbox/runtime/server.py": frozenset({"read_file"}),
    # "recall_past_sessions" here is the retained internal helper
    # recall_past_sessions_entries (memory(action='search',...) calls it),
    # same category as past_sessions.py's own exception below.
    "horizon/memory/add_memory_tool.py": frozenset(
        {"add_memory", "recall_past_sessions", "session_search"}
    ),
    "horizon/memory/_writer.py": frozenset({"add_memory"}),
    "horizon/memory/__init__.py": frozenset({"add_memory"}),
    "horizon/memory/review_fork.py": frozenset({"add_memory"}),
    "horizon/memory/flush_fork.py": frozenset({"add_memory"}),
    "horizon/builtin_skills/bootstrap-google-tools/SKILL.md": frozenset(
        {"read_file"}
    ),
    "docs/extending.md": frozenset({"read_file", "session_search"}),
    "docs/architecture.md": frozenset({"add_memory", "session_search"}),
    "docs/memory.md": frozenset({"add_memory"}),
    "docs/permission-model.md": frozenset({"session_search"}),
    "tests/eval/evalsets/dynamic_delegate_routing.evalset.json": frozenset(
        {"read_file"}
    ),
    "tests/eval/evalsets/memory_recall.evalset.json": frozenset({"add_memory"}),
    "tests/eval/evalsets/skill_curation.evalset.json": frozenset(
        {"add_memory"}
    ),
    "tests/eval/evalsets/tool_selection_core.evalset.json": frozenset(
        {"read_file", "add_memory"}
    ),
    "tests/eval/evalsets/workspace_window.evalset.json": frozenset(
        {"set_workspace_window"}
    ),
    # (d) the unrenamed add_memory_tool.py module filename, referenced by
    # name in the Project Layout tree; (g) historically-accurate prose
    # about set_workspace_window's deletion, same pattern as the evalset
    # exception above. Both surfaced only once README.md/AGENTS.md joined
    # the scan globs (final review Fix 8).
    "AGENTS.md": frozenset({"add_memory", "set_workspace_window"}),
    # session_search: historically-accurate prose about the merge into
    # memory(action='search'). horizon/tools/names.py's existing
    # frozenset(_LEGACY_TOKENS) entry above already covers its own
    # permanent TOOL_NAME_ALIASES key, so no second entry is needed there.
    "horizon/agent.py": frozenset({"session_search"}),
    "horizon/conversation/system_prompt.py": frozenset({"session_search"}),
    "horizon/guardrails/permission_guard.py": frozenset({"session_search"}),
    "horizon/subagents/delegate_builder.py": frozenset({"session_search"}),
    # old_string/replace_all: find_replacement's own kwargs, kept on purpose
    # (the model-facing error text now says oldText; the private helper's
    # signature did not change) — file_ops.py's and read.py's exceptions
    # above already cover their call sites. as_media: read.py's internal
    # _default_as_media dispatch helper, never a model-facing parameter
    # (also covered by read.py's entry above).
    "horizon/tools/_replacers.py": frozenset({"old_string", "replace_all"}),
}


async def test_no_legacy_tool_tokens_survive_after_the_rename():
    repo_root = _HORIZON_ROOT.parent
    offenders: dict[str, list[str]] = {}
    for glob in _LEGACY_SCAN_GLOBS:
        for path in repo_root.glob(glob):
            if not path.is_file():
                continue
            # web/**/*.ts(x) also globs vendored node_modules (4,289 of
            # 4,516 matches today) — never the intent, and the exact reason
            # "reload"/"patch"/"terminal" are unusable as scan tokens even
            # after excluding it (final review HIGH item).
            if "node_modules" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = str(path.relative_to(repo_root))
            allowed = _LEGACY_TOKEN_EXCEPTIONS.get(rel, frozenset())
            hits = [
                tok
                for tok in _LEGACY_TOKENS
                if tok in text and tok not in allowed
            ]
            if hits:
                offenders[rel] = hits
    assert not offenders, offenders


# Bare substrings for these four tokens are unusable as a scan (43/56/53/40+
# files each, dominated by unrelated meanings: web page/HMR/config reload,
# unittest.mock.patch, the web UI's own terminal/patch concepts, e2e test
# files). Scoping to the CALL shape — the token immediately followed by "("
# — collapses that to 4 real files, all legitimate: `terminal_exec.py`'s
# internal executor (still named `terminal`, distinct from the model-facing
# `bash` tool), `commands/__init__.py`'s internal `reload()` helper backing
# `/reload` (the model-facing action folded into `load_skill(action=
# 'reload')`; the Python function name did not change), and
# `Environment.write_file` in `environment/sandbox.py` +
# `sandbox/runtime/server.py` (the ADK BaseEnvironment method + its FastAPI
# route, never renamed — same exception class as `read_file` above).
# `delegate` is deliberately NOT in this scan: `subagents/delegate_runner.py`,
# `_delegate_resurfacing`, and `child_session_id = f"delegate-{fc_id}"` are
# all live, current code, so a call-shaped `delegate(` scan would need its
# own wide exception list rather than the 4 above — out of scope here.
_CALL_SHAPE_PATTERN = re.compile(
    r"(?<![\w.])(reload|patch|terminal|write_file)\("
)
_CALL_SHAPE_EXCEPTIONS: dict[str, frozenset[str]] = {
    "horizon/tools/terminal_exec.py": frozenset({"terminal"}),
    "horizon/commands/__init__.py": frozenset({"reload"}),
    "horizon/environment/sandbox.py": frozenset({"write_file"}),
    "horizon/sandbox/runtime/server.py": frozenset({"write_file"}),
}


async def test_no_legacy_tool_calls_survive_after_the_rename():
    """Would have caught the verification-review blockers a plain token scan
    missed: dead-tool-name PROSE survives easily (e.g. "the terminal tool"),
    but a dead-tool-name CALL ("terminal(...)", "patch(...)") is exactly the
    shape a model would try to invoke, and none should remain outside the
    four legitimate internal exceptions above.
    """
    repo_root = _HORIZON_ROOT.parent
    offenders: dict[str, list[str]] = {}
    for glob in (
        "horizon/**/*.py",
        "horizon/builtin_skills/**/*.md",
        "tests/eval/evalsets/*.json",
    ):
        for path in repo_root.glob(glob):
            if not path.is_file() or "node_modules" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = str(path.relative_to(repo_root))
            allowed = _CALL_SHAPE_EXCEPTIONS.get(rel, frozenset())
            hits = [
                m.group(1)
                for m in _CALL_SHAPE_PATTERN.finditer(text)
                if m.group(1) not in allowed
            ]
            if hits:
                offenders[rel] = hits
    assert not offenders, offenders
