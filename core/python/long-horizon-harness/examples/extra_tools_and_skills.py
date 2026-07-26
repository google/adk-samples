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

"""Extend Horizon with your own HTTP route, tool, and workspace skill.

Run it:

    uv run uvicorn examples.extra_tools_and_skills:app --port 8001

Three extension points — this is a sample, so you adapt it by editing the code:

* Routes — import the built app and ``app.include_router(...)`` your own
  ``APIRouter`` (shown below); it mounts alongside the built-in routes.
* Tools — add your function to the ``tools`` list in ``horizon/agent.py``. A
  plain typed function with a docstring is auto-wrapped as a FunctionTool; the
  docstring is what the model reads, so write it for the model. For example::

      def word_count(text: str) -> dict:
          \"\"\"Count the words in a piece of text.

          Args:
              text: The text to count words in.
          \"\"\"
          return {"words": len(text.split())}

* Skills — no code. Drop ``.agents/skills/<name>/SKILL.md`` (markdown how-to) in the
  agent's workspace; it's auto-discovered and ``/reload`` picks it up mid-session
  (see ../docs/extending.md §1).

Constructs offline — no GCP credentials. See ../docs/extending.md ("Add routers / tools / skills").
"""

import os

os.environ.setdefault("USE_IN_MEMORY_SESSION", "true")
os.environ.setdefault("LHA_ENVIRONMENT_BACKEND", "local")

from fastapi import APIRouter

from horizon.fast_api_app import app

health = APIRouter()


@health.get("/example/health")
def example_health() -> dict:
    return {"ok": True}


app.include_router(health)

__all__ = ["app"]
