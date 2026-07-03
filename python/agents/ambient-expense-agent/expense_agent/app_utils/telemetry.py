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

"""Telemetry setup for Agent Runtime deployment."""

import logging
import os


def setup_telemetry() -> None:
    """Configure basic telemetry env vars for Agent Runtime."""
    os.environ.setdefault("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "false")
    os.environ.setdefault("GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", "true")


def setup_agent_engine_telemetry() -> None:
    """Install Agent Engine tracer provider when running on Agent Runtime.

    No-op outside Agent Runtime (GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY
    not set), and silently skipped if vertexai package is not installed.
    """
    if os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", "").lower() not in (
        "true",
        "1",
    ):
        return
    try:
        import google.auth
        from vertexai.agent_engines.templates.adk import _default_instrumentor_builder

        _, project_id = google.auth.default()
        _default_instrumentor_builder(
            project_id, enable_tracing=True, enable_logging=True
        )
    except Exception as exc:
        logging.debug("Agent Engine telemetry not available: %s", exc)
