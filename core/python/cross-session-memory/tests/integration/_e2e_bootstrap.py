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

The subprocess uvicorn server does NOT load pytest conftest.py, so the
same test stubs applied there must be reapplied here BEFORE any app.*
module is imported. Also patches Gemini so no real Vertex AI call is
made from inside the subprocess.

Uvicorn imports this module as `_e2e_bootstrap:app`. test_server_e2e.py
extends the subprocess PYTHONPATH so both this file and the sibling
tests/_env_stubs.py are importable.
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Make sibling `tests/_env_stubs.py` importable in the subprocess.
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent))

from _env_stubs import install as _install_test_stubs  # noqa: E402

_install_test_stubs()

# Only NOW is it safe to import ADK models (they might touch env at import).
from google.adk.models.google_llm import Gemini  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402


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
