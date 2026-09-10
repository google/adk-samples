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

"""Unit tests for brand search optimization tools."""

from unittest.mock import MagicMock, patch

import pytest
from google.adk.tools.computer_use.base_computer import ComputerEnvironment

from brand_search_optimization.shared_libraries import constants
from brand_search_optimization.sub_agents.comparison.models import (
    TitleOptimizationReport,
    TitleRecommendation,
)
from brand_search_optimization.tools import bq_connector
from brand_search_optimization.tools.browser_computer import (
    MockBrowserComputer,
    get_browser_computer,
    get_computer_use_toolset,
)


class TestBigQueryConnector:
    """Tests for BigQuery catalog extraction tool."""

    @patch("brand_search_optimization.tools.bq_connector.client")
    def test_get_product_details_for_brand_success(self, mock_client):
        mock_row1 = MagicMock(
            Title="Cymbal Air Max",
            Description="Comfortable running shoes",
            Attributes="Size: 10, Color: Blue",
            Brand="Cymbal",
        )
        mock_row2 = MagicMock(
            Title="Cymbal Sportswear T-Shirt",
            Description="Cotton blend, short sleeve",
            Attributes="Size: L, Color: Black",
            Brand="Cymbal",
        )
        mock_results = [mock_row1, mock_row2]

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_results
        mock_client.query.return_value = mock_query_job

        with (
            patch.object(constants, "PROJECT", "test_project"),
            patch.object(constants, "TABLE_ID", "test_table"),
        ):
            response = bq_connector.get_product_details_for_brand(
                brand="Cymbal", limit=5
            )
            assert response.brand == "Cymbal"
            assert response.total_count == 2
            assert len(response.products) == 2
            assert response.products[0].title == "Cymbal Air Max"
            assert response.products[1].title == "Cymbal Sportswear T-Shirt"

    def test_get_product_details_for_brand_empty(self):
        response = bq_connector.get_product_details_for_brand(brand="")
        assert response.total_count == 0
        assert response.products == []

    @patch("brand_search_optimization.tools.bq_connector.client", None)
    @patch(
        "brand_search_optimization.tools.bq_connector._get_client",
        return_value=None,
    )
    def test_get_product_details_for_brand_offline_fallback(self, _):
        response = bq_connector.get_product_details_for_brand(brand="Acme")
        assert response.brand == "Acme"
        assert response.total_count >= 1
        assert any("Acme" in p.title for p in response.products)


class TestBrowserComputer:
    """Tests for Computer Use browser automation."""

    @pytest.mark.asyncio
    async def test_mock_browser_computer_operations(self):
        computer = MockBrowserComputer()

        size = await computer.screen_size()
        assert size == (1280, 800)

        env = await computer.environment()
        assert env == ComputerEnvironment.ENVIRONMENT_BROWSER

        state = await computer.open_web_browser()
        assert state.screenshot is not None
        assert len(state.screenshot) > 0

        nav_state = await computer.navigate(
            "https://www.google.com/search?tbm=shop&q=running+shoes"
        )
        assert "running+shoes" in nav_state.url

        type_state = await computer.type_text_at(
            x=100, y=200, text="kids sneakers"
        )
        assert "kids+sneakers" in type_state.url

        click_state = await computer.click_at(x=150, y=250)
        assert click_state.screenshot is not None

        scroll_state = await computer.scroll_document(direction="down")
        assert scroll_state.screenshot is not None

    def test_get_browser_computer_offline_flag(self):
        with patch.object(constants, "DISABLE_WEB_DRIVER", 1):
            computer = get_browser_computer()
            assert isinstance(computer, MockBrowserComputer)

    def test_get_computer_use_toolset(self):
        toolset = get_computer_use_toolset()
        assert toolset is not None


class TestModels:
    """Tests for Pydantic optimization schemas."""

    def test_title_optimization_report_schema(self):
        rec = TitleRecommendation(
            original_title="Runner",
            proposed_title="Brand Men's Breathable Mesh Pro Runner - Black Size 10",
            keywords_added=["Men's", "Breathable Mesh", "Pro Runner"],
            searchability_score=94.5,
            rationale="Captures gender, material, and category attributes.",
        )
        report = TitleOptimizationReport(
            brand="Brand",
            primary_search_keyword="running shoes",
            competitor_title_patterns=[
                "[Brand] [Gender] [Material] [Model] [Category]"
            ],
            keyword_gaps=["breathable mesh", "men's"],
            recommendations=[rec],
            summary_findings="Significant discovery score improvement expected.",
        )
        assert report.brand == "Brand"
        assert report.recommendations[0].searchability_score == 94.5
