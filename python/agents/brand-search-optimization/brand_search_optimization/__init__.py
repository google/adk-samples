# Copyright 2025 Google LLC
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
import types as builtin_types
import google.auth

import typing

typing._UnionGenericAlias = builtin_types.UnionType  # type: ignore[attr-defined]

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
if not project_id and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    _, project_id = google.auth.default()

os.environ.setdefault(
    "GOOGLE_CLOUD_PROJECT", project_id or "your-default-project"
)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

from .agent import root_agent

__all__ = ["root_agent"]
