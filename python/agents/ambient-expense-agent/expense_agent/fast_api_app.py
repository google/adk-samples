# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI entry point for the ambient expense agent backend.

Configures the ADK web server for Agent Runtime deployment with Pub/Sub
trigger endpoints enabled, allowing the agent to process expense reports
autonomously.

Trigger endpoint: POST /apps/expense_agent/trigger/pubsub
  - Receives Pub/Sub push messages
  - Decodes base64 payload and passes it as the agent's user input
  - Creates a session keyed by subscription name for tracking

Session storage: auto-detects ``GOOGLE_CLOUD_AGENT_ENGINE_ID`` (injected by
Agent Runtime) to use Vertex AI session service; falls back to in-memory
for local development.

Includes middleware to normalize Pub/Sub subscription names from their
fully-qualified resource paths (``projects/.../subscriptions/NAME``)
to short names, keeping user IDs clean and consistent with what the
frontend uses when querying for pending approvals.
"""

import json
import os

import uvicorn
from dotenv import load_dotenv
from google.adk.cli.fast_api import get_fast_api_app
from google.cloud import logging as google_cloud_logging
from starlette.requests import Request

from expense_agent.app_utils import services
from expense_agent.app_utils.telemetry import (
    setup_agent_engine_telemetry,
    setup_telemetry,
)
from expense_agent.app_utils.typing import Feedback

load_dotenv()
setup_telemetry()
# Must run before get_fast_api_app to set the tracer provider resource.
setup_agent_engine_telemetry()
logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=False,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=False,
    trigger_sources=["pubsub"],  # exposes /apps/expense_agent/trigger/pubsub
)
app.title = "ambient-expense-agent"
app.description = "Ambient expense agent — processes expense reports via Pub/Sub"


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback."""
    logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


@app.middleware("http")
async def normalize_pubsub_subscription(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Normalize ``projects/.../subscriptions/NAME`` to just ``NAME``.

    Pub/Sub push deliveries include the fully-qualified subscription
    resource path. The ADK trigger handler uses this value as the
    session ``user_id``. Normalizing to the short name keeps session
    records clean and consistent with the subscription name used by
    the frontend when querying for pending approvals.
    """
    if request.url.path.endswith("/trigger/pubsub") and request.method == "POST":
        body = await request.body()
        try:
            data = json.loads(body)
            sub = data.get("subscription", "")
            if "/" in sub:
                data["subscription"] = sub.rsplit("/", 1)[-1]
                request._body = json.dumps(data).encode()
        except (json.JSONDecodeError, KeyError):
            pass
    return await call_next(request)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
    )
