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

"""Unit tests for tools"""

from unittest.mock import MagicMock, patch

from google.adk.tools import ToolContext

from brand_search_optimization.tools import bq_connector
from brand_search_optimization.shared_libraries import constants


class TestBrandSearchOptimization:

    @patch("brand_search_optimization.tools.bq_connector.client")
    def test_get_product_details_for_brand_success(self, mock_client):
        # Mock ToolContext
        mock_tool_context = MagicMock(spec=ToolContext)
        mock_tool_context.user_content.parts = [MagicMock(text="cymbal")]

        # Mock BigQuery results
        mock_row1 = MagicMock(
            title="cymbal Air Max",
            description="Comfortable running shoes",
            attribute="Size: 10, Color: Blue",
            brand="cymbal",
        )
        mock_row2 = MagicMock(
            title="cymbal Sportswear T-Shirt",
            description="Cotton blend, short sleeve",
            attribute="Size: L, Color: Black",
            brand="cymbal",
        )
        mock_row3 = MagicMock(
            title="neuravibe Pro Training Shorts",
            description="Moisture-wicking fabric",
            attribute="Size: M, Color: Gray",
            brand="neuravibe",
        )
        mock_results = [mock_row1, mock_row2, mock_row3]

        # Mock QueryJob and its result
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_results
        mock_client.query.return_value = mock_query_job

        # Mock constants
        with patch.object(constants, "PROJECT", "test_project"):
            with patch.object(constants, "TABLE_ID", "test_table"):
                # Call the function
                markdown_output = bq_connector.get_product_details_for_brand(
                    mock_tool_context
                )
                assert "neuravibe Pro" not in markdown_output
