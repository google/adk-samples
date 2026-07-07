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

import google.auth
from google.adk import Context, Event
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.workflow import node
from google.adk.workflow._node_runner import NodeRunner
from google.genai import types

# ==============================================================================
# Initialize the environment
# ==============================================================================
_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# ==============================================================================
# Model Definition
# ==============================================================================
model = Gemini(
    model="gemini-3.5-flash",
    retry_options=types.HttpRetryOptions(attempts=3),
)

# ==============================================================================
# Sub-Agent Definition
# ==============================================================================
task_explainer = LlmAgent(
    name="task_explainer",
    model=model,
    instruction="""
    You are a task execution planner subagent.
    Given a task description, write a short, step-by-step execution plan
    explaining how you would perform the task. Be concise and clear.
    """,
)


# ==============================================================================
# Dynamic Workflow Nodes
# ==============================================================================
@node(rerun_on_resume=True)
async def tasks_workflow(ctx: Context, node_input: list[str]):
    """Dynamic workflow that iterates over the list of tasks."""
    for task in node_input:
        # Yield progress update
        yield Event(message=f"⏳ Starting task: {task}...")  # type: ignore

        # Dynamically trigger subagent
        explanation = await ctx.run_node(task_explainer, node_input=task)

        # Extract explanation text from types.Content or other types
        if hasattr(explanation, "text") and explanation.text:
            explanation_content = explanation.text
        elif hasattr(explanation, "parts") and explanation.parts:
            explanation_content = "".join(
                [p.text for p in explanation.parts if p.text]
            )
        else:
            explanation_content = str(explanation)

        # Mark as done
        yield Event(
            message=(
                f"✅ Task Done: {task}\n\n"
                f"**Execution Explanation:**\n{explanation_content}"
            )  # type: ignore
        )

    yield Event(output="🎉🚀 All tasks executed successfully! ✨")


# ==============================================================================
# Tools
# ==============================================================================
async def run_workflow(ctx: Context, tasks: list[str]) -> str:
    """Executes the provided list of tasks using the dynamic workflow.

    Args:
        tasks: A list of tasks to execute.
    """
    runner = NodeRunner(
        node=tasks_workflow,
        parent_ctx=ctx,
    )
    child_ctx = await runner.run(node_input=tasks)

    if child_ctx.error:
        return f"Error: Task execution failed.\n{child_ctx.error}"

    return child_ctx.output or "All tasks completed successfully."


# ==============================================================================
# Root Agent & App
# ==============================================================================
root_agent = LlmAgent(
    name="root_agent",
    model=model,
    instruction="""
    You are a task coordinator agent.
    Your goal is to gather a list of tasks that the user wants to execute.
    Talk to the user to gather the list of tasks.
    Once you have a list of tasks, present them clearly to the user and ask
    for their final approval to execute them.
    Do NOT execute anything until the user explicitly approves.
    Once the user approves the list of tasks, call the tool `run_workflow` with
    the list of tasks.
    """,
    tools=[run_workflow],
)

app = App(
    root_agent=root_agent,
    name="app",
)
