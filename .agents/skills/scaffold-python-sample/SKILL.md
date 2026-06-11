---
name: scaffold-python-sample
description: >
  This skill should be used when the user wants to "create a new Python ADK sample",
  "scaffold a new Python sample project", "generate a new Python sample in contrib",
  "add a new Python sample to the adk-samples repository", or "create a Python adk sample".
  It generates a new prototype agent project directly under the `contrib/` directory
  of the `adk-samples` repository by writing files and folders directly.
metadata:
  author: Google
  license: Apache-2.0
  version: 0.2.0
---

# Scaffolding a New Python ADK Sample

Use this skill to create and scaffold a new Python ADK sample project inside the repository. This skill creates all necessary files and folders directly.

---

## Step 1: Clarify Requirements

Before scaffolding the project, ensure you have the following information from the user:
1. **Project Name** (Required): The name of the new sample project.
   - **Constraints**: Must be **26 characters or less**, lowercase letters, numbers, and hyphens only.
2. **Google AI Studio API Key** (Optional): A `GEMINI_API_KEY` to use Google AI Studio instead of Vertex AI.
3. **Output Directory** (Optional): The directory inside the repository where the project should be created.
   - **Default**: `contrib/`
   - **Constraints**: Must be a relative path inside the repository (e.g., `contrib/`, `samples/`, or `python/`).

If the project name does not meet the constraints, ask the user for clarification before proceeding.

---

## Core Rules for Template Adherence

When generating the files under `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/`, you **MUST STRICTLY** adhere to the following rules:
1. **No Structural Modifications**: Do **NOT** alter, omit, or modify the structure, headers, fields, or metadata defined in the templates below. The generated files must be an **exact, literal copy** of the templates.
2. **Placeholder Replacement Only**: Only replace the designated placeholders (like `<PROJECT_NAME>`, `<RECIPE TITLE HERE>`, `<BRIEF DESCRIPTION HERE>`, etc.) with concrete values.
3. **No Field Deletion**: Do **NOT** remove any metadata fields, comments, or configuration blocks (such as `Point of Contact`, telemetry configuration, or warning filters) even if they seem redundant or you think they are not needed for a specific sample.
4. **Consistency**: Ensure that if a template has a field, it is present in the final generated file exactly as written.

---

## Step 2: Generate Files and Folders

Create the following files and folders under `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/` in the repository.
- **`<OUTPUT_DIRECTORY>`**: Defaults to `contrib/` unless the user explicitly specified a different directory in Step 1.
- For all the file paths listed below, replace `contrib/` with the user-specified `<OUTPUT_DIRECTORY>` if a different one was provided.

### 1. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/pyproject.toml`
```toml
[project]
name = "<PROJECT_NAME>"
version = "0.1.0"
description = ""
dependencies = [
    "google-adk>=2.0.0",
    "python-dotenv>=1.0.0",
    "opentelemetry-instrumentation-google-genai>=0.1.0,<1.0.0",
    "opentelemetry-exporter-otlp-proto-http>=1.26.0",
    "opentelemetry-exporter-gcp-logging>=1.12.0a0",
    "gcsfs>=2024.11.0",
    "google-cloud-logging>=3.12.0,<4.0.0",
]
requires-python = "3.11"

[dependency-groups]
dev = [
    "pytest>=9.0.2,<10.0.0",
    "pytest-asyncio>=1.0.0,<2.0.0",
    "nest-asyncio>=1.6.0,<2.0.0",
]

[project.optional-dependencies]
eval = [
    "google-adk[eval]>=2.0.0",
]
lint = [
    "ruff>=0.4.6,<1.0.0",
    "ty>=0.0.1a0",
    "codespell>=2.2.0,<3.0.0",
]

[tool.ruff]
line-length = 80
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle
    "F",   # pyflakes
    "W",   # pycodestyle warnings
    "I",   # isort
    "C",  # flake8-comprehensions
    "B",   # flake8-bugbear
    "UP", # pyupgrade
    "RUF", # ruff specific rules
]
ignore = ["E501", "C901", "B006"] # ignore line too long, too complex

[tool.ruff.lint.isort]
known-first-party = ["app", "frontend"]

[tool.ty]
# ty is Astral's Rust-based type checker (same team as ruff/uv)
# See: https://docs.astral.sh/ty/

[tool.ty.environment]
python-version = "3.11"

[tool.ty.src]
exclude = [".venv/**"]

[tool.ty.rules]
# Ignore common issues with third-party libraries and dynamic code patterns
unresolved-import = "ignore"
unresolved-attribute = "ignore"
invalid-argument-type = "ignore"
invalid-assignment = "ignore"
invalid-return-type = "ignore"
possibly-missing-attribute = "ignore"
not-subscriptable = "ignore"
deprecated = "ignore"

[tool.codespell]
ignore-words-list = "rouge"
skip = "./locust_env/*,uv.lock,.venv,./frontend,**/package-lock.json"


[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"


[tool.pytest.ini_options]
pythonpath = "."
asyncio_default_fixture_loop_scope = "function"

[tool.hatch.build.targets.wheel]
packages = ["app","frontend"]
```

### 2. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/GEMINI.md`
```markdown
# Coding Agent Guide

## Prerequisites

Install uv (one-time):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Development Phases

### Phase 1: Understand Requirements
Before writing any code, understand the project's requirements, constraints, and success criteria.

### Phase 2: Build and Implement
Implement agent logic in `app/`. Iterate based on user feedback.

### Phase 3: Pre-Deployment Tests
Run `uv run pytest tests/unit tests/integration`. Fix issues until all tests pass.

## Operational Guidelines for Coding Agents

- **Code preservation**: Only modify code directly targeted by the user's request. Preserve all surrounding code, config values (e.g., `model`), comments, and formatting.
- **NEVER change the model** unless explicitly asked.
- **Model 404 errors**: Fix `GOOGLE_CLOUD_LOCATION` (e.g., `global` instead of `us-east1`), not the model name.
- **ADK tool imports**: Import the tool instance, not the module: `from google.adk.tools.load_web_page import load_web_page`
- **Run Python with `uv`**: `uv run python script.py`.
- **Stop on repeated errors**: If the same error appears 3+ times, fix the root cause instead of retrying.
```

### 3. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/README.md`
```markdown
# <PROJECT_NAME>

Simple ADK agent
This is a simple agent using the ADK Python SDK to demonstrate its capabilities. 

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager - [Install](https://docs.astral.sh/uv/getting-started/installation/)

## Quick Start

1. Install required packages:
   ```bash
   uv sync
   ```

2. Set up environment variables:
   Copy `.env.example` to `.env` and uncomment/configure the variables you need (like `GEMINI_API_KEY`, `GOOGLE_CLOUD_PROJECT`, etc.):
   ```bash
   cp .env.example .env
   ```

3. Test the agent in the command line (interactive mode):
   ```bash
   uv run adk run app
   ```

4. Or start the local FastAPI web server:
   ```bash
   uv run uvicorn app.fast_api_app:app --reload
   ```

## Running Tests

To run the unit, integration, and runnability tests:

```bash
uv run pytest
```

Or to run specific test suites:

```bash
# Run unit and runnability tests only
uv run pytest tests/unit

# Run integration tests only
uv run pytest tests/integration
```

## Commands

| Command | Description |
| ------- | ----------- |
| `uv run adk run app` | Run the agent in interactive CLI mode |
| `uv run uvicorn app.fast_api_app:app --reload` | Start the local FastAPI development server |
| `uv run pytest` | Run all test suites |
```

### 4. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/app/__init__.py`
```python
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

from .agent import app

__all__ = ["app"]
```

### 5. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/app/agent.py`
```python
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

import datetime
import os
from zoneinfo import ZoneInfo

import google.auth
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

# Load the .env file
load_dotenv()

<VERTEX_AI_ENV_SETUP>

def get_weather(query: str) -> str:
    """Simulates a web search. Use it get information on weather.

    Args:
        query: A string containing the location to get weather information for.

    Returns:
        A string with the simulated weather information for the queried location.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


def get_current_time(query: str) -> str:
    """Simulates getting the current time for a city.

    Args:
        city: The name of the city to get the current time for.

    Returns:
        A string with the current time information.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        tz_identifier = "America/Los_Angeles"
    else:
        return f"Sorry, I don't have timezone information for query: {query}."

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return f"The current time for query {query} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model=os.getenv("MODEL_NAME", "gemini-flash-latest"),
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="You are a helpful AI assistant designed to provide accurate and useful information.",
    tools=[get_weather, get_current_time],
)

app = App(
    root_agent=root_agent,
    name="app",
)
```
*(Note: If `GEMINI_API_KEY` was NOT provided, replace `<VERTEX_AI_ENV_SETUP>` with the following block. If it WAS provided, replace `<VERTEX_AI_ENV_SETUP>` with an empty string)*

**Vertex AI Env Setup Block:**
```python
import os
import google.auth

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
```

### 6. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/app/.env` (Only if `GEMINI_API_KEY` is provided)
```env
# AI Studio Configuration
GEMINI_API_KEY=<GEMINI_API_KEY>
```

### 7. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/app/fast_api_app.py`
```python
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
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.cloud import logging as google_cloud_logging

from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback

# Load the .env file
load_dotenv()

setup_telemetry()
    _, project_id = google.auth.default()
    logging_client = google_cloud_logging.Client()
    logger = logging_client.logger(__name__)
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

# Artifact bucket for ADK (created by Terraform, passed via env var)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# In-memory session configuration - no persistent storage
session_service_uri = None

artifact_service_uri = f"gs://{logs_bucket_name}" if logs_bucket_name else None

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=artifact_service_uri,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
    otel_to_cloud=True,
)
app.title = "<PROJECT_NAME>"
app.description = "API for interacting with the Agent <PROJECT_NAME>"


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
        logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 8. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/app/app_utils/telemetry.py`
```python
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

import logging
import os


def setup_telemetry() -> str | None:
    """Configure OpenTelemetry and GenAI telemetry with GCS upload."""

    bucket = os.environ.get("LOGS_BUCKET_NAME")
    capture_content = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false"
    )
    if bucket and capture_content != "false":
        logging.info(
            "Prompt-response logging enabled - mode: NO_CONTENT (metadata only, no prompts/responses)"
        )
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "NO_CONTENT"
        os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_UPLOAD_FORMAT", "jsonl")
        os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK", "upload")
        os.environ.setdefault(
            "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
        )
        commit_sha = os.environ.get("COMMIT_SHA", "dev")
        os.environ.setdefault(
            "OTEL_RESOURCE_ATTRIBUTES",
            f"service.namespace=<PROJECT_NAME>,service.version={commit_sha}",
        )
        path = os.environ.get("GENAI_TELEMETRY_PATH", "completions")
        os.environ.setdefault(
            "OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH",
            f"gs://{bucket}/{path}",
        )
    else:
        logging.info(
            "Prompt-response logging disabled (set LOGS_BUCKET_NAME=gs://your-bucket and OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT to enable)"
        )

    return bucket
```

### 9. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/app/app_utils/typing.py`
```python
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

import uuid
from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


class Feedback(BaseModel):
    """Represents feedback for a conversation."""

    score: int | float
    text: str | None = ""
    log_type: Literal["feedback"] = "feedback"
    service_name: Literal["<PROJECT_NAME>"] = "<PROJECT_NAME>"
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
```

### 10. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/app/app_utils/__init__.py`
*(Create an empty file)*

### 11. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/tests/unit/test_tools.py`
```python
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
"""Unit tests for the custom weather and time tools."""

from app.agent import get_current_time, get_weather


def test_get_weather_san_francisco() -> None:
    """Tests that get_weather returns foggy weather for San Francisco."""
    result = get_weather("San Francisco")
    assert "60 degrees and foggy" in result


def test_get_weather_other_city() -> None:
    """Tests that get_weather returns sunny weather for other cities."""
    result = get_weather("London")
    assert "90 degrees and sunny" in result


def test_get_current_time_san_francisco() -> None:
    """Tests that get_current_time returns time for San Francisco."""
    result = get_current_time("San Francisco")
    assert "America/Los_Angeles" in result or "PDT" in result or "PST" in result


def test_get_current_time_unknown_city() -> None:
    """Tests that get_current_time returns error message for unknown cities."""
    result = get_current_time("London")
    assert "Sorry, I don't have timezone information" in result
```

### 12. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/tests/unit/test_runnability.py`
```python
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
"""Runnability tests for the ADK agent sample project.

Verifies that the python code compiles, resolves all dependencies, and imports
correctly under zero-configuration CI/CD conditions without throwing errors.
"""

import os

from fastapi import FastAPI
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini


def test_fast_api_app_runnability() -> None:
    """Verifies fast_api_app.py compiles and initializes FastAPI successfully."""
    # Importing fast_api_app triggers the telemetry and mock global setups
    import app.fast_api_app

    assert app.fast_api_app.app is not None
    assert isinstance(app.fast_api_app.app, FastAPI)
    assert app.fast_api_app.app.title == "<PROJECT_NAME>"


def test_agent_runnability() -> None:
    """Verifies agent.py compiles and instantiates the agent flow successfully."""
    # Importing agent loads the tools and the Agent runner configuration
    import app.agent

    # 1. Assert ADK App is initialized correctly
    assert app.agent.app is not None
    assert isinstance(app.agent.app, App)
    assert app.agent.app.name == "app"

    # 2. Assert Agent and its properties are built with matching parameters
    assert app.agent.root_agent is not None
    assert isinstance(app.agent.root_agent, Agent)
    assert app.agent.root_agent.name == "root_agent"
    assert isinstance(app.agent.root_agent.model, Gemini)
    assert app.agent.root_agent.model.model == os.getenv(
        "MODEL_NAME", "gemini-flash-latest"
    )

    # 3. Assert Tools set contains expected analytical tools
    tools = app.agent.root_agent.tools
    assert len(tools) == 2

    # Find registered helper functions
    tool_names = [getattr(t, "__name__", type(t).__name__) for t in tools]
    assert "get_weather" in tool_names
    assert "get_current_time" in tool_names
```

### 13. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/tests/integration/test_agent.py`
```python
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

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


def test_agent_stream() -> None:
    """
    Integration test for the agent stream functionality.
    Tests that the agent returns valid streaming responses.
    """

    session_service = InMemorySessionService()

    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    message = types.Content(
        role="user", parts=[types.Part.from_text(text="Why is the sky blue?")]
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one message"

    has_text_content = False
    for event in events:
        if (
            event.content
            and event.content.parts
            and any(part.text for part in event.content.parts)
        ):
            has_text_content = True
            break
    assert has_text_content, "Expected at least one message with text content"
```

### 14. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/tests/integration/test_server_e2e.py`
```python
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

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
import requests
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8000"
STREAM_URL = BASE_URL + "/run_sse"
FEEDBACK_URL = BASE_URL + "/feedback"

HEADERS = {"Content-Type": "application/json"}


def log_output(pipe: Any, log_func: Any) -> None:
    """Log the output from the given pipe."""
    for line in iter(pipe.readline, ""):
        log_func(line.strip())


def start_server() -> subprocess.Popen[str]:
    """Start the FastAPI server using subprocess and log its output."""
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.fast_api_app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    env = os.environ.copy()
    env["INTEGRATION_TEST"] = "TRUE"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    # Start threads to log stdout and stderr in real-time
    threading.Thread(
        target=log_output, args=(process.stdout, logger.info), daemon=True
    ).start()
    threading.Thread(
        target=log_output, args=(process.stderr, logger.error), daemon=True
    ).start()

    return process


def wait_for_server(timeout: int = 90, interval: int = 1) -> bool:
    """Wait for the server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://127.0.0.1:8000/docs", timeout=10)
            if response.status_code == 200:
                logger.info("Server is ready")
                return True
        except RequestException:
            pass
        time.sleep(interval)
    logger.error(f"Server did not become ready within {timeout} seconds")
    return False


@pytest.fixture(scope="session")
def server_fixture(request: Any) -> Iterator[subprocess.Popen[str]]:
    """Pytest fixture to start and stop the server for testing."""
    logger.info("Starting server process")
    server_process = start_server()
    if not wait_for_server():
        pytest.fail("Server failed to start")
    logger.info("Server process started")

    def stop_server() -> None:
        logger.info("Stopping server process")
        server_process.terminate()
        server_process.wait()
        logger.info("Server process stopped")

    request.addfinalizer(stop_server)
    yield server_process


def test_chat_stream(server_fixture: subprocess.Popen[str]) -> None:
    """Test the chat stream functionality."""
    logger.info("Starting chat stream test")
    # Create session first
    user_id = "test_user_123"
    session_data = {"state": {"preferred_language": "English", "visit_count": 1}}

    session_url = f"{BASE_URL}/apps/app/users/{user_id}/sessions"
    session_response = requests.post(
        session_url,
        headers=HEADERS,
        json=session_data,
        timeout=60,
    )
    assert session_response.status_code == 200
    logger.info(f"Session creation response: {session_response.json()}")
    session_id = session_response.json()["id"]

    # Then send chat message
    data = {
        "app_name": "app",
        "user_id": user_id,
        "session_id": session_id,
        "new_message": {
            "role": "user",
            "parts": [{"text": "Hi!"}],
        },
        "streaming": True,
    }
    response = requests.post(
        STREAM_URL, headers=HEADERS, json=data, stream=True, timeout=60
    )
    assert response.status_code == 200

    # Parse SSE events from response
    events = []
    for line in response.iter_lines():
        if line:
            # SSE format is "data: {json}"
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                event_json = line_str[6:]  # Remove "data: " prefix
                event = json.loads(event_json)
                events.append(event)

    assert events, "No events received from stream"
    # Check for valid content in the response
    has_text_content = False
    for event in events:
        content = event.get("content")
        if (
            content is not None
            and content.get("parts")
            and any(part.get("text") for part in content["parts"])
        ):
            has_text_content = True
            break

    assert has_text_content, "Expected at least one event with text content"


def test_chat_stream_error_handling(server_fixture: subprocess.Popen[str]) -> None:
    """Test the chat stream error handling."""
    logger.info("Starting chat stream error handling test")
    data = {
        "input": {"messages": [{"type": "invalid_type", "content": "Cause an error"}]}
    }
    response = requests.post(
        STREAM_URL, headers=HEADERS, json=data, stream=True, timeout=10
    )

    assert response.status_code == 422, (
        f"Expected status code 422, got {response.status_code}"
    )
    logger.info("Error handling test completed successfully")


def test_collect_feedback(server_fixture: subprocess.Popen[str]) -> None:
    """
    Test the feedback collection endpoint (/feedback) to ensure it properly
    logs the received feedback.
    """
    # Create sample feedback data
    feedback_data = {
        "score": 4,
        "user_id": "test-user-456",
        "session_id": "test-session-456",
        "text": "Great response!",
    }

    response = requests.post(
        FEEDBACK_URL, json=feedback_data, headers=HEADERS, timeout=10
    )
    assert response.status_code == 200
```

### 15. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/tests/eval/eval_config.json`
```json
{
  "criteria": {
    "rubric_based_final_response_quality_v1": {
      "threshold": 0.8,
      "judgeModelOptions": {
        "judgeModel": "gemini-flash-latest",
        "numSamples": 1
      },
      "rubrics": [
        {
          "rubricId": "relevance",
          "rubricContent": { "textProperty": "The response directly addresses the user's query." }
        },
        {
          "rubricId": "helpfulness",
          "rubricContent": { "textProperty": "The response is helpful and provides useful information." }
        }
      ]
    }
  }
}
```

### 16. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/tests/eval/evalsets/README.md`
```markdown
# Evaluation Sets

This directory contains evaluation sets for testing agent behavior.

## Evalset Format

Each `.evalset.json` follows the ADK evaluation format:

```json
{
  "eval_set_id": "unique_id",
  "name": "Human-readable name",
  "description": "What this evalset tests",
  "eval_cases": [
    {
      "eval_id": "case_id",
      "conversation": [
        {
          "user_content": {
            "parts": [{"text": "User message"}]
          }
        }
      ],
      "session_input": {
        "app_name": "app_name",
        "user_id": "test_user",
        "state": {}
      }
    }
  ]
}
```
```

### 17. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/tests/eval/evalsets/basic.evalset.json`
```json
{
  "eval_set_id": "basic_eval",
  "name": "Basic Agent Evaluation",
  "description": "Sample evaluation set for testing core agent functionality. Customize these cases for your agent.",
  "eval_cases": [
    {
      "eval_id": "greeting",
      "conversation": [
        {
          "user_content": {
            "parts": [{"text": "Hello, what can you help me with?"}]
          }
        }
      ],
      "session_input": {
        "app_name": "app",
        "user_id": "eval_user",
        "state": {}
      }
    },
    {
      "eval_id": "weather_query",
      "conversation": [
        {
          "user_content": {
            "parts": [{"text": "What's the weather like in San Francisco?"}]
          }
        }
      ],
      "session_input": {
        "app_name": "app",
        "user_id": "eval_user",
        "state": {}
      }
    }
  ]
}
```

### 18. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/.env.example`

```env
# Model Configuration
MODEL_NAME=gemini-flash-latest

# Google Cloud Platform Configuration (for Vertex AI)
# GOOGLE_CLOUD_PROJECT=your-gcp-project-id
# GOOGLE_CLOUD_LOCATION=global
# GOOGLE_GENAI_USE_VERTEXAI=True

# Google AI Studio Configuration (if using API Key instead of Vertex AI)
# GEMINI_API_KEY=your-api-key-here

# Telemetry and Logging Configuration
# LOGS_BUCKET_NAME=your-gcs-bucket-name
# OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT

# Web Server Configuration
# ALLOW_ORIGINS=http://localhost:3000,http://localhost:8080
```

### 19. `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/RECIPE.md`

```markdown
# <RECIPE TITLE HERE>
<BRIEF DESCRIPTION HERE>

- **Recipe Type**: pattern
- **Owner**: <TEAM NAME HERE> (<TEAM EMAIL HERE>)
- **Point of Contact**: <POINT OF CONTACT NAME> | <POINT OF CONTACT EMAIL>
- **Status**: active
- **Tags**: <COMMA SEPARATED TAGS>
- **Languages**: python, <COMMA SEPARATED LANGUAGES>
- **Paired Skill**: `scaffold-python-sample`
- **Evaluation**: `tests/eval/evalsets/basic.evalset.json` (min score: 0.8)

## Intent

It highlights a simple ADK agent with 2 python tools.

## When To Use

- To scaffoled a simple ADK agent.

## Requires
A GCP Project if the user wants to deploy it to Google Cloud

## Constraints

- **Must**: user interaction, Gemini Enterprise
- **Must Not**: service-account-only, no UI

## Composition

- **Composes with**: `agent-runtime-deployment`
- **Conflicts with**: None
```

---

## Step 3: Post-Scaffold Verification and Next Steps

Once all files are successfully written:
1. Verify that the project was created under `<OUTPUT_DIRECTORY>/<PROJECT_NAME>`.
2. Inform the user that the project is ready and point them to `<OUTPUT_DIRECTORY>/<PROJECT_NAME>/README.md`.
3. Inform the user they can get started by navigating to the project directory and running:
   - To run the agent in the command line (interactive mode):
     ```bash
     cd <OUTPUT_DIRECTORY>/<PROJECT_NAME> && uv sync && uv run adk run app
     ```
   - To start the FastAPI server:
     ```bash
     cd <OUTPUT_DIRECTORY>/<PROJECT_NAME> && uv sync && uv run uvicorn app.fast_api_app:app --reload
     ```
