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

import pytest
from customer_service.config import Config
from dotenv import find_dotenv, load_dotenv
from google.adk.evaluation.agent_evaluator import AgentEvaluator

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv(find_dotenv(".env"))
    c = Config()


@pytest.mark.asyncio
async def test_eval_simple():
    """Test the agent's basic ability via a session file."""
    await AgentEvaluator.evaluate(
        "customer_service",
        os.path.join(os.path.dirname(__file__), "eval_data/simple.test.json"),
        num_runs=1,
    )


@pytest.mark.asyncio
async def test_eval_full_conversation():
    """Test the agent's basic ability via a session file."""
    await AgentEvaluator.evaluate(
        "customer_service",
        os.path.join(
            os.path.dirname(__file__), "eval_data/full_conversation.test.json"
        ),
        num_runs=1,
    )
