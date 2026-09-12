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

import contextlib
import os
from collections.abc import AsyncIterator

from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner

from financial_advisor.app_utils import services
from financial_advisor.app_utils.a2a import attach_a2a_routes
from financial_advisor.app_utils.reasoning_engine_adapter import (
    attach_reasoning_engine_routes,
)

load_dotenv()
allow_origins = (
    [
        origin.strip()
        for origin in os.getenv("ALLOW_ORIGINS").split(",")
        if origin.strip()
    ]
    if os.getenv("ALLOW_ORIGINS")
    else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _otel_to_cloud_enabled() -> bool:
    """Whether to export ADK's own traces/metrics to Cloud Trace/Monitoring.

    Cloud export requires GCP credentials / metadata server. When deployed on
    Cloud Run (which sets `K_SERVICE`), telemetry is enabled by default.
    For local development, it defaults to false to avoid 30s metadata server
    timeouts. Set `OTEL_TO_CLOUD=true` to override.
    """
    override = os.getenv("OTEL_TO_CLOUD")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes"}
    return bool(os.getenv("K_SERVICE"))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from financial_advisor.agent import app as adk_app
    from financial_advisor.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    task_store = InMemoryTaskStore()
    app.state.runner = runner
    app.state.task_store = task_store
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=task_store,
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=_otel_to_cloud_enabled(),
    lifespan=lifespan,
)
app.title = "financial-advisor"
app.description = "API for interacting with the Agent financial-advisor"
attach_reasoning_engine_routes(app)


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/.well-known/agent-card.json", include_in_schema=False)
async def root_agent_card() -> RedirectResponse:
    return RedirectResponse(
        url="/a2a/financial_advisor/.well-known/agent-card.json"
    )


# Main execution
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_graceful_shutdown=10)
