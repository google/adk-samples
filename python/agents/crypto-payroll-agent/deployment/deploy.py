"""Deploy the Crypto Payroll Agent to Vertex AI Agent Engine.

From the agent directory:
    uv run python deployment/deploy.py
"""

from __future__ import annotations

import os
import sys

import vertexai
from vertexai.preview import reasoning_engines

# Make the package importable from the repo root.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from crypto_payroll_agent import root_agent  # noqa: E402
from crypto_payroll_agent.config import CONFIG  # noqa: E402


def main() -> None:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    staging_bucket = os.environ.get(
        "GOOGLE_CLOUD_STAGING_BUCKET", f"gs://{project}-adk-staging"
    )

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
        requirements=[
            "google-adk>=1.0.0",
            (
                "google-adk-community @ "
                "git+https://github.com/google/adk-python-community.git@main"
            ),
            "google-cloud-aiplatform[adk,agent_engines]>=1.95.0",
            "web3>=6.0",
            "python-dotenv>=1.0",
        ],
        extra_packages=["./crypto_payroll_agent"],
    )

    print(f"Deployed: {remote_app.resource_name}")


if __name__ == "__main__":
    main()
