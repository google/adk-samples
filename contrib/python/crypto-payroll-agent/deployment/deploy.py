"""Deploy the Crypto Payroll Agent to Vertex AI Agent Engine.

From the agent directory:
    uv run python deployment/deploy.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import tomllib
import vertexai
from vertexai.preview import reasoning_engines

# Make the package importable from the repo root.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from crypto_payroll_agent import root_agent
from crypto_payroll_agent.config import CONFIG


def _requirements() -> list[str]:
    """Read the engine's requirements from the recipe's pyproject.toml.

    Read rather than restated, so the deployed engine cannot drift from
    what uv.lock resolves and the tests cover — including the exact
    commit the google-adk-community dependency is pinned to.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        return tomllib.load(f)["project"]["dependencies"]


def main() -> None:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ["GOOGLE_CLOUD_LOCATION"]
    staging_bucket = os.environ["GOOGLE_CLOUD_STAGING_BUCKET"]

    vertexai.init(
        project=project, location=location, staging_bucket=staging_bucket
    )

    app = reasoning_engines.AdkApp(agent=root_agent, enable_tracing=True)

    remote_app = vertexai.agent_engines.create(
        agent_engine=app,
        display_name=CONFIG.app_name,
        description=(
            "Crypto Payroll Agent — batch stablecoin and ETH payouts on "
            "Base via the Spraay community tools."
        ),
        requirements=_requirements(),
        extra_packages=["./crypto_payroll_agent"],
        # config.py raises on a missing PAYROLL_* variable, and the engine
        # reads it at import time — without these the deployed agent fails
        # to start.
        #
        # SPRAAY_PRIVATE_KEY is deliberately absent: the Spraay tools need
        # it at tool-call time, but an env var here is readable by anyone
        # who can describe the engine, and this key controls real funds.
        # Wire it up from Secret Manager before deploying for real.
        env_vars={
            "PAYROLL_AGENT_MODEL": os.environ["PAYROLL_AGENT_MODEL"],
            "PAYROLL_MAX_BATCH_USD": os.environ["PAYROLL_MAX_BATCH_USD"],
            "PAYROLL_MAX_BATCH_ETH": os.environ["PAYROLL_MAX_BATCH_ETH"],
        },
    )

    print(f"Deployed: {remote_app.resource_name}")


if __name__ == "__main__":
    main()
