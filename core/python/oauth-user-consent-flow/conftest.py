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

# Skip integration tests unless INTEGRATION_TEST is explicitly set.
# Integration tests import the agent package at module level, which
# requires MODEL_NAME and live GCP credentials — neither of which is
# available in a plain `uv run pytest` run without a populated .env.
collect_ignore_glob: list[str] = []
if not os.environ.get("INTEGRATION_TEST"):
    collect_ignore_glob.append("tests/integration/*.py")
