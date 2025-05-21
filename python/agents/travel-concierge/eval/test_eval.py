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

"""Basic evaluation of the travel concierge agent."""

import pathlib

import dotenv
from google.adk.evaluation import AgentEvaluator
import pytest


@pytest.fixture(scope="session", autouse=True)
def load_env():
    dotenv.load_dotenv()


def test_inspire():
    """Test the agent's basic ability on a few examples."""
    AgentEvaluator.evaluate(
        "travel_concierge",
        str(pathlib.Path(__file__).parent / "data/inspire.test.json"),
        num_runs=4,
        initial_session_file=str(pathlib.Path(__file__).parent
                                 / "itinerary_empty_default.json")
    )


def test_pretrip():
    """Test the agent's basic ability on a few examples."""
    AgentEvaluator.evaluate(
        "travel_concierge",
        str(pathlib.Path(__file__).parent / "data/pretrip.test.json"),
        num_runs=4,
        initial_session_file=str(pathlib.Path(__file__).parent
                                 / "itinerary_seattle_example.json")
    )


def test_intrip():
    """Test the agent's basic ability on a few examples."""
    AgentEvaluator.evaluate(
        "travel_concierge",
        str(pathlib.Path(__file__).parent / "data/intrip.test.json"),
        num_runs=4,
        initial_session_file=str(pathlib.Path(__file__).parent
                                 / "itinerary_seattle_example.json")
    )
    