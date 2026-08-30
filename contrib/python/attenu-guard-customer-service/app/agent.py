# Copyright 2026 Attenu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Customer-service coordinator with a billing sub-agent, guarded.

Two agents, one plugin. `DelegationGuardPlugin` is registered once on the
`App`; from then on every agent in the tree carries a permission set, and
every tool call is checked against it before the tool body runs.

The coordinator can issue a refund. The billing agent it transfers to
cannot: it is delegated `billing.read` and nothing else, so its
`issue_refund` call is refused at the callback, and the function is never
entered. The refusal is appended to a hash-chained ledger that verifies
without this process, this recipe, or any network.
"""

import os
from typing import Any

from attenu_guard import Guard
from attenu_guard.adapters.google_adk import DelegationGuardPlugin
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App

from .permissions import (
    COORDINATOR,
    DELEGATION_SCOPE,
    DELEGATIONS,
    TOOLS,
)
from .prompt import BILLING_PROMPT, COORDINATOR_PROMPT
from .tools import email_customer, get_invoice, issue_refund, lookup_order

APP_NAME = "attenu-guard-customer-service"
ROOT_AGENT_NAME = "coordinator"


def build_root_agent(model: Any) -> LlmAgent:
    """The agent tree. `model` is a model name or a `BaseLlm` instance,
    so the offline demo can pass a scripted model in its place."""
    billing_agent = LlmAgent(
        name="billing_agent",
        model=model,
        description="Reads invoices and explains billing questions.",
        instruction=BILLING_PROMPT,
        tools=[get_invoice, issue_refund, email_customer],
    )
    return LlmAgent(
        name=ROOT_AGENT_NAME,
        model=model,
        description="Handles customer-service requests end to end.",
        instruction=COORDINATOR_PROMPT,
        tools=[lookup_order],
        sub_agents=[billing_agent],
    )


def build_plugin(root_guard: Guard) -> DelegationGuardPlugin:
    """One plugin covers the whole tree."""
    return DelegationGuardPlugin(
        root_guard,
        root_agent_name=ROOT_AGENT_NAME,
        delegations=DELEGATIONS,
        tools=TOOLS,
        delegation_scope=DELEGATION_SCOPE,
    )


def build_app(
    model: Any,
    *,
    task: str = "customer service",
    audit_path: str | None = None,
) -> tuple[App, Guard, DelegationGuardPlugin]:
    """An `App` with the guard attached, plus the root `Guard` and the
    plugin. You keep the root Guard: it owns the ledger, the delegation
    graph, and `revoke()`."""
    root_guard = Guard.issue(
        ROOT_AGENT_NAME, COORDINATOR, task=task, audit_path=audit_path
    )
    plugin = build_plugin(root_guard)
    application = App(
        name=APP_NAME,
        root_agent=build_root_agent(model),
        plugins=[plugin],
    )
    require_guard(application)
    return application, root_guard, plugin


def require_guard(application: App) -> None:
    """Refuse to run an App that lost its guard.

    Cheap, and it turns "somebody removed the plugin" from a silent
    downgrade into a startup failure.
    """
    plugins = getattr(application, "plugins", None) or []
    if not any(isinstance(p, DelegationGuardPlugin) for p in plugins):
        raise RuntimeError(
            "DelegationGuardPlugin is not attached to this App — "
            "refusing to run unguarded"
        )


# The module-level objects the ADK CLI looks for. `adk run` and `adk web`
# check for `app` first and fall back to `root_agent`, so exposing the App
# is what carries the plugin into a CLI-driven run.
#
# One caveat worth knowing: these are built once, at import, so every
# session `adk web` serves shares this root Guard and this ledger, and the
# time-to-live starts counting at import. That is fine for trying the
# recipe out. For anything where one caller's decisions should not be
# another's evidence, call `build_app()` per run — which is what `demo.py`
# and the tests do.
app, root_guard, guard_plugin = build_app(os.getenv("MODEL_NAME", ""))
root_agent = app.root_agent
