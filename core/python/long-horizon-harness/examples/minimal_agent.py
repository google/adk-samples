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

"""Smallest runnable Horizon harness: Gemini, tools on the host, in-memory everything.

Run it:

    uv run uvicorn examples.minimal_agent:app --port 8001

Then drive it over A2A at http://127.0.0.1:8001/a2a, or read /lha/sessions.

Horizon is configured entirely by environment variable — this file just sets the
lowest-friction combo before importing the served app (every router mounts; edit
`horizon/fast_api_app.py` to trim). Constructs offline — no GCP credentials, no
Cloud SQL / Agent Engine. Inference still calls Vertex per request (there is no
local model). See ../docs/quickstart.md, ../docs/configuration.md, ../docs/extending.md.
"""

import os

os.environ.setdefault("USE_IN_MEMORY_SESSION", "true")  # no Cloud SQL / Agent Engine
os.environ.setdefault("LHA_ENVIRONMENT_BACKEND", "local")  # tools run on this host
os.environ.setdefault(
    "LHA_ROOT_MODEL", "gemini-3.6-flash"
)  # Gemini needs no Model Garden step

from horizon.fast_api_app import app

__all__ = ["app"]
