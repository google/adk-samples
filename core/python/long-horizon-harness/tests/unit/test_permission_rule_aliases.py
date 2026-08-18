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

"""The tool-name alias layer must not open a security hole.

A rule persisted under a legacy tool name (an overlay file, a session
grant) has to keep resolving to the live tool after a rename, without
letting the aliasing step itself defeat the untrusted-blanket-allow
narrowing check that guards shell tools.

Two of these tests deviate from the plan's original snippets, which called
functions that do not exist (``policies_guard_verdict``) or asserted
against the wrong entry point (``parse_rule(...) is None``, which conflicts
with existing tests in ``tests/unit/guardrails/test_permission_rules.py``
that rely on ``parse_rule({"toolName": "terminal", "decision": "allow"})``
returning a real rule for hand-built ``resolve_decision`` fixtures). See
the corrected versions below and the delegation report for details.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from horizon.guardrails import permission_rules as pr
from horizon.guardrails.permission_guard import resolve_permission_decision
from horizon.guardrails.permission_rules import parse_rule
from horizon.guardrails.policies import policies_guard
from horizon.tools import names

pytestmark = pytest.mark.asyncio


def test_legacy_rule_still_matches_current_tool():
    rule = parse_rule(
        {
            "toolName": "terminal",
            "commandPrefix": ["bq rm"],
            "decision": "allow",
        },
        trusted=False,
    )
    assert rule is not None
    assert names.BASH in rule.tool_names


def test_aliased_blanket_allow_is_still_dropped():
    # The plan's original snippet asserted `parse_rule(...) is None`, but
    # parse_rule never applies the blanket-allow check on its own -- only
    # `_stamp` does, and only for overlay/grant-sourced rules
    # (effective_rules is the real enforcement point). Existing tests in
    # test_permission_rules.py rely on parse_rule({"toolName": "terminal",
    # "decision": "allow"}) returning a real rule (they build resolve_decision
    # fixtures by hand), so making parse_rule itself return None here would
    # break them. This test instead exercises the real enforcement path:
    # _TOOLS_REQUIRING_NARROWING gates _is_blanket_allow, which _stamp uses
    # to drop an untrusted blanket allow. If aliasing rewrote "terminal" to
    # a new live name without also updating _TOOLS_REQUIRING_NARROWING, the
    # rule would stop being recognized as one that needs narrowing and would
    # survive _stamp -- a blanket allow for shell, unnarrowed, from an
    # untrusted source.
    rule = parse_rule({"toolName": "terminal", "decision": "allow"})
    assert rule is not None
    assert names.BASH in rule.tool_names
    assert pr._is_blanket_allow(rule)
    stamped = pr._stamp([rule], "overlay")
    assert stamped == []


async def test_hard_deny_floor_survives_aliasing():
    # dd if=... matches the JSONL destructive_commands substring list
    # directly (policies.py's _evaluate), independent of command_safety.
    blocked = await policies_guard(
        tool=SimpleNamespace(name=names.BASH),
        args={"command": "dd if=/dev/zero of=/dev/sda"},
        tool_context=SimpleNamespace(),
    )
    assert blocked is not None and "error" in blocked


async def test_command_safety_deny_still_fires_for_shell():
    # The plan's original snippet used a fork-bomb command
    # (":(){ :|:& };:", with spaces) intending to exercise
    # classify_command's deny tier. That exact string is not what
    # command_safety.classify recognizes (it returns None for it, with or
    # without spaces -- only the JSONL seed's no-space substring
    # ":(){:|:&};:" catches a fork bomb, via _evaluate, not classify_command).
    # "rm -rf /" isolates classify_command's own deny path instead: it is
    # not in the JSONL substring list, and classify_command("rm -rf /")
    # returns ("deny", "recursive force-delete of a system/home root")
    # (verified directly against horizon.guardrails.command_safety.classify).
    # If policies._command_for stopped recognizing names.BASH, this call
    # would return None (allow) instead of a block.
    blocked = await policies_guard(
        tool=SimpleNamespace(name=names.BASH),
        args={"command": "rm -rf /"},
        tool_context=SimpleNamespace(),
    )
    assert blocked is not None and "error" in blocked


async def test_command_substitution_still_demotes_to_ask():
    # resolve_permission_decision(env, *, tool_name, args, state, agent_name)
    # is async and returns (decision, deny_rule, proposed_prefix, command),
    # not a single object with a `.decision` attribute as the plan's
    # original snippet assumed.
    decision, _deny, _prefix, _cmd = await resolve_permission_decision(
        None,
        tool_name=names.BASH,
        args={"command": "echo $(cat /etc/passwd)"},
        state={},
        agent_name="root_agent",
    )
    assert decision == "ask_user"
