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

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env if present. In production the environment is
# already populated by the platform (Cloud Run, GKE, etc.), so a missing
# .env is expected and not an error.
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.is_file():
    load_dotenv(_env_path)
else:
    load_dotenv()

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

from .agent import root_agent  # noqa: E402 -- must come after load_dotenv()

__all__ = ["root_agent"]
