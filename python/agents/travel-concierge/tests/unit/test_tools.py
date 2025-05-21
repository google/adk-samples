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

"""Basic tests for individual tools."""

import unittest

from dotenv import load_dotenv
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
import pytest
from travel_concierge.agent import root_agent
from travel_concierge.tools.memory import memorize
from travel_concierge.tools.places import map_tool


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()


session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()


class TestAgents(unittest.TestCase):
    """Test cases for the Travel Concierge cohort of agents."""

    def setUp(self):
        """Set up for test methods."""
        super().setUp()
        self.session = session_service.create_session(
            app_name="Travel_Concierge",
            user_id="traveler0115",
        )
        self.user_id = "traveler0115"
        self.session_id = self.session.id

        self.invoc_context = InvocationContext(
            session_service=session_service,
            invocation_id="ABCD",
            agent=root_agent,
            session=self.session,
        )
        self.tool_context = ToolContext(invocation_context=self.invoc_context)

    def test_memory(self):
        result = memorize(
            key="itinerary_datetime",
            value="12/31/2025 11:59:59",
            tool_context=self.tool_context,
        )
        self.assertIn("status", result)
        self.assertEqual(
            self.tool_context.state["itinerary_datetime"], "12/31/2025 11:59:59"
        )

    def test_places(self):
        self.tool_context.state["poi"] = {
            "places": [{"place_name": "Machu Picchu", "address": "Machu Picchu, Peru"}]
        }
        result = map_tool(key="poi", tool_context=self.tool_context)
        print(result)
        self.assertIn("place_id", result[0])
        self.assertEqual(
            self.tool_context.state["poi"]["places"][0]["place_id"],
            "ChIJVVVViV-abZERJxqgpA43EDo",
        )
