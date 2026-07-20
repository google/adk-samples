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

"""Test-only ASGI entrypoint for the e2e server subprocess.

Patches the Gemini LLM with a canned response before importing the app so
the subprocess never makes a real Vertex AI call. Uvicorn imports this
module as `_e2e_bootstrap:app` (the subprocess PYTHONPATH is extended by
test_server_e2e.py to include this directory).
"""

from unittest.mock import patch

from google.adk.models.google_llm import Gemini
from google.adk.models.llm_response import LlmResponse
from google.genai import types


async def _fake_generate_content(self, llm_request, stream=False):
    """Mock LLM that returns a canned text response without any API call."""
    yield LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Hello from the mock LLM.")],
        ),
        turn_complete=True,
    )


# Apply the patch for the lifetime of this process, before importing the app.
patch.object(Gemini, "generate_content_async", _fake_generate_content).start()

# Now safe to import the app -- any Gemini() built inside will use the mock.
from app.fast_api_app import app  # noqa: E402

__all__ = ["app"]
