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

"""Session and artifact services shared across ADK serving surfaces.

On Agent Runtime, ``GOOGLE_CLOUD_AGENT_ENGINE_ID`` is auto-injected by the
platform; the session service switches to ``VertexAiSessionService`` so that
sessions persist across replicas and are queryable via the API passthrough.
Falls back to in-memory for local development.
"""

from __future__ import annotations

import functools
import os

from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService

# Use shared:// URIs so the ADK web routes and any future A2A path share
# one session/artifact service instance per process.
SESSION_SERVICE_URI = "shared://session"
ARTIFACT_SERVICE_URI = "shared://artifact"

@functools.cache
def get_session_service():
    """Return a session service, using Vertex AI on Agent Runtime."""
    if agent_engine_id := os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
        from google.adk.sessions.vertex_ai_session_service import (
            VertexAiSessionService,
        )

        return VertexAiSessionService(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=(
                os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
                or os.environ.get("GOOGLE_CLOUD_LOCATION")
            ),
            agent_engine_id=agent_engine_id,
        )
    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    return InMemorySessionService()


@functools.cache
def get_artifact_service():
    """GCS artifact service when LOGS_BUCKET_NAME is set, else in-memory."""
    if bucket := os.environ.get("LOGS_BUCKET_NAME"):
        return GcsArtifactService(bucket_name=bucket)
    return InMemoryArtifactService()


try:
    from google.adk.cli.service_registry import get_service_registry

    _registry = get_service_registry()
    _registry.register_session_service(
        "shared", lambda uri, **kw: get_session_service()
    )
    _registry.register_artifact_service(
        "shared", lambda uri, **kw: get_artifact_service()
    )
except Exception:
    pass  # service_registry is optional; sessions still work via env vars
