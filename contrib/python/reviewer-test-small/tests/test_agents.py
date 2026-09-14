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

"""Test cases for the Task Planner Agent"""

import textwrap

import dotenv
import pytest
from google.adk.runners import InMemoryRunner
from google.genai.types import Part, UserContent

from sdlc_task_planner.agent import root_agent

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_happy_path():
    """Runs the agent on a simple input and expects a normal response."""
    user_input = textwrap.dedent(
        """Here is the user story and technical design document:
        User Story: As a user, I want to be able to reset my password so that I can regain access to my account if I forget it.

        Technical Design:
        1. Create a new endpoint `/api/forgot-password` that accepts an email address and sends a password reset link.
        2. Create a new endpoint `/api/reset-password` that accepts a reset token and a new password, and updates the user's password in the database.
        3. Database: Add a `reset_token` and `reset_token_expires_at` column to the `users` table.
    """
    ).strip()

    runner = InMemoryRunner(agent=root_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="test_user"
    )
    content = UserContent(parts=[Part(text=user_input)])
    response = ""
    artifact_content = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=content,
    ):
        print(event)
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response = part.text
                if (
                    part.function_call
                    and part.function_call.name == "save_artifact"
                ):
                    if part.function_call.args:
                        artifact_content = part.function_call.args.get(
                            "content", ""
                        )

    assert "Task ID" in artifact_content
    assert "Technical Description & Files" in artifact_content
    assert "Acceptance Criteria & Testing" in artifact_content
    assert response != ""
