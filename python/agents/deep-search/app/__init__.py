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

"""Deep Search Agent: strategize, research, and synthesize comprehensive reports."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env if present. app/config.py also loads this file,
# but the mode check below runs before that import.
load_dotenv(Path(__file__).parent / ".env")

# AI Studio mode (GOOGLE_API_KEY set) runs without Cloud credentials, so the
# project id lookup only applies to Vertex AI. Matches app/config.py.
if not os.getenv("GOOGLE_API_KEY"):
    import google.auth

    _, project_id = google.auth.default()
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id or "")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

from app.agent import root_agent
