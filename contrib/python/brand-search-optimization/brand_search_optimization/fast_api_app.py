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
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner

from brand_search_optimization.app_utils.a2a import attach_a2a_routes
from brand_search_optimization.app_utils.services import (
    AGENT_DIR,
    ARTIFACT_SERVICE_URI,
    SESSION_SERVICE_URI,
    get_artifact_service,
    get_session_service,
)

allow_origins = (
    [
        origin.strip()
        for origin in os.getenv("ALLOW_ORIGINS").split(",")
        if origin.strip()
    ]
    if os.getenv("ALLOW_ORIGINS")
    else None
)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from brand_search_optimization.agent import app as adk_app
    from brand_search_optimization.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=get_session_service(),
        artifact_service=get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=SESSION_SERVICE_URI,
    otel_to_cloud=True,
    lifespan=lifespan,
)
app.title = "brand-search-optimization"
app.description = "API for interacting with the Agent brand-search-optimization"


# Main execution
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST")
    port_str = os.getenv("PORT")
    port = int(port_str) if port_str else None
    uvicorn.run(app, host=host, port=port)
