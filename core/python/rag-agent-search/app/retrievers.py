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

import os
from collections.abc import Callable

from google.adk.tools import VertexAiSearchTool


def create_search_tool(
    data_store_path: str,
) -> VertexAiSearchTool | Callable[[str], str]:
    """Create a Agent Platform Search tool or mock for testing.

    Args:
        data_store_path: Full resource path of the datastore.

    Returns:
        VertexAiSearchTool instance or mock function for testing.
    """
    # For integration tests, return a mock function instead of the real tool
    if os.getenv("INTEGRATION_TEST") == "TRUE":

        def mock_search(query: str) -> str:
            """Mock Agent Platform Search for integration tests."""
            return "Mock search result for testing purposes."

        return mock_search

    return VertexAiSearchTool(data_store_id=data_store_path)
