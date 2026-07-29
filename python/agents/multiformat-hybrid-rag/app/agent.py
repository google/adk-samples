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
import sys

import vertexai
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import (
    McpToolset,
    SseConnectionParams,
    StdioConnectionParams,
)
from google.genai import types
from mcp import StdioServerParameters

from app.config import AGENT_MODEL, LOCATION, PROJECT_ID

os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

LLM = AGENT_MODEL
LLM_LOCATION = "global" if "preview" in LLM else LOCATION
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")

os.environ["GOOGLE_CLOUD_LOCATION"] = LLM_LOCATION

vertexai.init(project=PROJECT_ID, location=LOCATION)

# MCP connection: SSE if MCP_SERVER_URL is set, otherwise stdio (subprocess)
if MCP_SERVER_URL:
    mcp_connection = SseConnectionParams(url=MCP_SERVER_URL, timeout=120)
else:
    mcp_connection = StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp_server"],
        ),
        timeout=120,
    )

instruction = """\
You are a knowledge base assistant. Your role is to help users find accurate \
information from the indexed documents.

Rules:
- ALWAYS use the ask_knowledge_base tool before answering. Pass a summary of \
the conversation and the user's question. The tool searches the knowledge base \
and returns an answer grounded in the documents.
- Use the tool's response as the basis for your answer. You may rephrase or \
restructure for clarity, but do not add information beyond what was provided.
- Reply in the same language as the user.
- Be concise and direct.
- If a question is ambiguous, ask for clarification rather than guessing."""


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model=LLM,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=instruction,
    tools=[McpToolset(connection_params=mcp_connection)],
)

app = App(
    root_agent=root_agent,
    name="app",
)
