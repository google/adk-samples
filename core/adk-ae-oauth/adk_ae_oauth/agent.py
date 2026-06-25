# ruff: noqa
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

import os

import google.auth
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from adk_ae_oauth.tools import read_drive_file

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model="gemini-3-flash-preview",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are a helpful AI assistant that can read files from Google Drive.

When a user wants to read a file from Google Drive:
1. Ask for the file ID if they haven't provided one. The file ID is the
   alphanumeric string found in the Google Drive sharing URL, for example:
   https://drive.google.com/file/d/<FILE_ID>/view
2. Use the read_drive_file tool with the file ID.
3. Present the file content to the user in a clear, readable format.
4. If the tool returns a "pending" status, let the user know that
   authentication is required and they should complete the OAuth consent.

You can also help users understand or summarise the content once it is loaded.
""",
    tools=[read_drive_file],
)

app = App(
    root_agent=root_agent,
    name="app",
)
