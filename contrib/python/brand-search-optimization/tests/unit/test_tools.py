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

"""Unit tests for Brand Search Optimization tools and ADK BaseComputer Playwright."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.adk.tools.computer_use.base_computer import (
    ComputerEnvironment,
)

from brand_search_optimization.sub_agents.search_results.agent import (
    search_results_agent,
)
from brand_search_optimization.sub_agents.search_results.playwright_computer import (
    PlaywrightComputer,
    _format_url,
)
from brand_search_optimization.tools import bq_connector


class TestBrandSearchOptimization:
    @patch("brand_search_optimization.tools.bq_connector.client")
    def test_get_product_details_for_brand_success(self, mock_client):
        mock_row1 = MagicMock(
            Title="cymbal Air Max",
            Description="Comfortable running shoes",
            Attributes="Size: 10, Color: Blue",
            Brand="cymbal",
        )
        mock_row2 = MagicMock(
            Title="cymbal Sportswear T-Shirt",
            Description="Cotton blend, short sleeve",
            Attributes="Size: L, Color: Black",
            Brand="cymbal",
        )
        mock_row3 = MagicMock(
            Title="neuravibe Pro Training Shorts",
            Description="Moisture-wicking fabric",
            Attributes="Size: M, Color: Gray",
            Brand="neuravibe",
        )
        mock_results = [mock_row1, mock_row2, mock_row3]

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_results
        mock_client.query.return_value = mock_query_job

        with patch.dict(
            "os.environ",
            {"GOOGLE_CLOUD_PROJECT": "test_project", "TABLE_ID": "test_table"},
        ):
            markdown_output = bq_connector.get_product_details_for_brand(
                brand_name="cymbal"
            )
            assert "cymbal Air Max" in markdown_output

    def test_format_url_valid(self):
        url = _format_url("google.com/search?q=shoes")
        assert url == "https://google.com/search?q=shoes"

    def test_format_url_blocks_unsafe_schemes(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            _format_url("file:///etc/passwd")
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            _format_url("javascript:alert(1)")

    @pytest.mark.asyncio
    async def test_playwright_base_computer_methods(self):
        comp = PlaywrightComputer(screen_size=(1440, 900))
        assert await comp.screen_size() == (1440, 900)
        assert (
            await comp.environment() == ComputerEnvironment.ENVIRONMENT_BROWSER
        )

        mock_page = AsyncMock()
        mock_page.url = "https://www.google.com"
        mock_page.screenshot.return_value = b"fake_png_bytes"
        comp._page = mock_page

        state = await comp.open_web_browser()
        assert state.url == "https://www.google.com"
        assert state.screenshot == b"fake_png_bytes"
        mock_page.goto.assert_awaited_once()

        click_state = await comp.click_at(100, 200)
        assert click_state.url == "https://www.google.com"
        mock_page.mouse.click.assert_awaited_once_with(100, 200)

        type_state = await comp.type_text_at(
            100, 200, "running shoes", press_enter=True
        )
        assert type_state.url == "https://www.google.com"
        mock_page.keyboard.type.assert_awaited_once_with("running shoes")

        scroll_state = await comp.scroll_document("down")
        assert scroll_state.url == "https://www.google.com"

        hover_state = await comp.hover_at(50, 50)
        assert hover_state.url == "https://www.google.com"
        mock_page.mouse.move.assert_awaited_with(50, 50)

        nav_state = await comp.navigate("https://www.bing.com")
        assert nav_state.url == "https://www.google.com"

    def test_search_results_agent_toolset(self):
        assert search_results_agent.name == "search_results_agent"
        assert len(search_results_agent.tools) > 0
