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

"""
Utility for ERA data types.
"""

# pylint: disable=unsupported-binary-operation
import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class HumanMessage(BaseModel):
    content: str
    type: str = "human"


class AIMessage(BaseModel):
    content: str
    type: str = "ai"


class ToolMessage(BaseModel):
    content: str
    tool_call_id: str
    type: str = "tool"


class InputChat(BaseModel):
    """Represents the input for a chat session."""

    messages: list[HumanMessage | AIMessage | ToolMessage] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


class Request(BaseModel):
    """Represents the input for a chat request with optional configuration.

    Attributes:
        input: The chat input containing messages and other chat-related data
        config: Optional configuration for the runnable, including tags,
        callbacks.
    """

    input: InputChat
    config: dict | None = None


class Feedback(BaseModel):
    """Represents feedback for a conversation."""

    score: int | float
    text: str | None = ""
    run_id: str
    log_type: Literal["feedback"] = "feedback"


def ensure_valid_config(config: dict | None) -> dict:
    """Ensures a valid RunnableConfig by setting defaults for missing fields."""
    if config is None:
        config = {}
    if config.get("run_id") is None:
        config["run_id"] = uuid.uuid4()
    if config.get("metadata") is None:
        config["metadata"] = {}
    return config


def default_serialization(obj: Any) -> Any:
    """
    Default serialization for LangChain objects.
    Converts BaseModel instances to JSON strings.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()
    return None


def dumps(obj: Any) -> str:
    """
    Serialize an object to a JSON string.
    """
    return json.dumps(obj, default=default_serialization)
