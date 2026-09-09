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

"""Unit tests for the SDLC Workflow Suite configuration, prompts, and agent structure."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext

from sdlc_workflow_suite.agent import (
    root_agent,
    task_planner_agent,
    technical_designer_agent,
    user_story_refiner_agent,
)
from sdlc_workflow_suite.config import AgentConfig
from sdlc_workflow_suite.prompt import (
    _get_spanner_config_instruction,
    get_task_planner_prompt,
    get_technical_designer_prompt,
    get_user_story_refiner_prompt,
)
from sdlc_workflow_suite.tools.artifact_tools import save_artifact
from sdlc_workflow_suite.tools.spanner_query_tools import SpannerQueryTools


def test_agent_config_defaults():
    """Test that the agent config provides standard defaults and aliases."""
    cfg = AgentConfig()
    assert cfg.model_name is not None
    assert cfg.default_llm == cfg.model_name
    assert cfg.spanner_project_id is None


def test_user_story_refiner_prompt():
    """Test user story refiner prompt with and without Spanner tools."""
    prompt_with_tools = get_user_story_refiner_prompt(tools_enabled=True)
    assert "Context & Knowledge Base Retrieval" in prompt_with_tools
    assert (
        "Actively query Spanner to retrieve relevant context"
        in prompt_with_tools
    )
    assert "Context Limitations" not in prompt_with_tools

    prompt_without_tools = get_user_story_refiner_prompt(tools_enabled=False)
    assert "Context Limitations" in prompt_without_tools
    assert (
        "You do NOT have access to search tools or external databases"
        in prompt_without_tools
    )
    assert "Context & Knowledge Base Retrieval" not in prompt_without_tools


def test_technical_designer_prompt():
    """Test technical designer prompt with and without Spanner tools."""
    prompt_with_tools = get_technical_designer_prompt(tools_enabled=True)
    assert "Instructions for Context Retrieval" in prompt_with_tools
    assert "Code Knowledge Graph" in prompt_with_tools
    assert "Context Limitations" not in prompt_with_tools

    prompt_without_tools = get_technical_designer_prompt(tools_enabled=False)
    assert "Context Limitations" in prompt_without_tools
    assert "Instructions for Context Retrieval" not in prompt_without_tools


def test_task_planner_prompt():
    """Test task planner prompt content and required format sections."""
    prompt = get_task_planner_prompt()
    assert "Comprehensive Task Table" in prompt
    assert "Task ID" in prompt
    assert "Technical Description & Files" in prompt
    assert "Acceptance Criteria & Testing" in prompt


def test_agent_composition():
    """Verify that root_agent is a SequentialAgent with the 3 sub-agents in order."""
    assert isinstance(root_agent, SequentialAgent)
    assert len(root_agent.sub_agents) == 3
    assert root_agent.sub_agents[0] == user_story_refiner_agent
    assert root_agent.sub_agents[1] == technical_designer_agent
    assert root_agent.sub_agents[2] == task_planner_agent


def test_spanner_config_instruction():
    """Test that the shared Spanner config instruction is included in prompts."""
    instruction = _get_spanner_config_instruction()
    assert "When using Spanner tools" in instruction
    assert instruction in get_user_story_refiner_prompt(tools_enabled=True)
    assert instruction in get_technical_designer_prompt(tools_enabled=True)


def test_task_planner_has_save_artifact():
    """Verify task planner has the save_artifact tool registered."""
    assert save_artifact in task_planner_agent.tools


def test_spanner_query_tools_unconfigured():
    """Verify get_toolset returns an empty list when Spanner config is unset."""
    assert SpannerQueryTools.get_toolset() == []


@pytest.mark.asyncio
async def test_save_artifact_tool():
    """Test save_artifact tool executes and returns formatted response."""
    mock_context = MagicMock(spec=ToolContext)
    mock_context.save_artifact = AsyncMock(return_value=1)

    result = await save_artifact(
        tool_context=mock_context,
        content="# Plan\nTasks here",
        filename="execution_plan",
    )
    assert result["status"] == "success"
    assert result["filename"] == "execution_plan.md"
    assert result["version"] == 1
    mock_context.save_artifact.assert_awaited_once()
