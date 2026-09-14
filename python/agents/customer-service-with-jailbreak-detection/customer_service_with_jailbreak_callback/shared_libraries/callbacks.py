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

"""Callback functions for Customer Service Agent with Jailbreak Detection."""

import logging
import time
from typing import Any, Dict, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.tools import BaseTool
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google import genai

from customer_service_with_jailbreak_callback.entities.customer import Customer
from customer_service_with_jailbreak_callback import prompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

client = genai.Client()

RATE_LIMIT_SECS = 60
RPM_QUOTA = 10
MODEL_NAME = "gemini-2.0-flash-001"


def rate_limit_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Callback function that implements a query rate limit.

    Args:
      callback_context: A CallbackContext obj representing the active callback
        context.
      llm_request: A LlmRequest obj representing the active LLM request.
    """
    for content in llm_request.contents:
        for part in content.parts:
            if part.text=="":
                part.text=" "


    now = time.time()
    if "timer_start" not in callback_context.state:

        callback_context.state["timer_start"] = now
        callback_context.state["request_count"] = 1
        logger.debug(
            "rate_limit_callback [timestamp: %i, "
            "req_count: 1, elapsed_secs: 0]",
            now,
        )
        return

    request_count = callback_context.state["request_count"] + 1
    elapsed_secs = now - callback_context.state["timer_start"]
    logger.debug(
        "rate_limit_callback [timestamp: %i, request_count: %i,"
        " elapsed_secs: %i]",
        now,
        request_count,
        elapsed_secs,
    )

    if request_count > RPM_QUOTA:
        delay = RATE_LIMIT_SECS - elapsed_secs + 1
        if delay > 0:
            logger.debug("Sleeping for %i seconds", delay)
            time.sleep(delay)
        callback_context.state["timer_start"] = now
        callback_context.state["request_count"] = 1
    else:
        callback_context.state["request_count"] = request_count

    return


def lowercase_value(value):
    """Make dictionary lowercase"""
    if isinstance(value, dict):
        return (dict(k, lowercase_value(v)) for k, v in value.items())
    elif isinstance(value, str):
        return value.lower()
    elif isinstance(value, (list, set, tuple)):
        tp = type(value)
        return tp(lowercase_value(i) for i in value)
    else:
        return value


# Callback Methods
def before_tool(
    tool: BaseTool, args: Dict[str, Any], tool_context: CallbackContext
):

    # i make sure all values that the agent is sending to tools are lowercase
    lowercase_value(args)

    # Check for the next tool call and then act accordingly.
    # Example logic based on the tool being called.
    if tool.name == "sync_ask_for_approval":
        amount = args.get("value", None)
        if amount <= 10:  # Example business rule
            return {
                "result": "You can approve this discount; no manager needed."
            }
        # Add more logic checks here as needed for your tools.

    if tool.name == "modify_cart":
        if (
            args.get("items_added") is True
            and args.get("items_removed") is True
        ):
            return {"result": "I have added and removed the requested items."}
    return None


# checking that the customer profile is loaded as state.
def before_agent(callback_context: InvocationContext):
    if "customer_profile" not in callback_context.state:
        callback_context.state["customer_profile"] = Customer.get_customer(
            "123"
        ).to_json()

    # logger.info(callback_context.state["customer_profile"])


def classify_user_input(text: str) -> int:

    print(f"[Classifier LLM] Analyzing text: '{text}'")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=prompt.JAILBREAK_FILTER_INSTRUCTION,
                max_output_tokens=2
            ),
            contents=text
        )
        classifier_output_text = response.text.replace(
            "'", "").replace('"', "").strip().lower()
        print(f"[Classifier LLM] Raw output: '{classifier_output_text}'")

        if "yes" in classifier_output_text:
            print(
                "[Classifier LLM] Parsed output: 1 "
                "(Return predefined response)"
            )
            return 1
        elif "no" in classifier_output_text:
            print("[Classifier LLM] Parsed output: 0 (Proceed with agent)")
            return 0
        else:
            print(
                "[Classifier LLM] Warning: Unexpected output "
                f"'{classifier_output_text}'. Defaulting to 0."
            )
            return 0 # Default to passing to agent if output is not as expected

    except Exception as e:
        print(f"[Classifier LLM] Error during classification: {e}")
        # Fallback behavior: if classification fails, the main agent handles it
        return 0


def routing_before_agent_callback(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """
    A before-agent callback that classifies user input for jailbreak attempts.

    If a jailbreak is detected:
    1. Sets a flag `is_jailbreak_attempt` to True in the session state.
    2. Halts the main agent's execution by returning a predefined Content object.

    If no jailbreak is detected:
    1. Sets the flag to False.
    2. Returns None, allowing the main agent to proceed normally.
    """
    agent_input = callback_context.user_content
    user_text = ""
    if agent_input and agent_input.parts and agent_input.parts[0].text:
        user_text = agent_input.parts[0].text
    else:
        print(
            f"[Callback:{callback_context.agent_name}] "
            "No text found in agent_input. Allowing agent to proceed."
        )
        return None
    print(
        f"[Callback:{callback_context.agent_name}] Intercepted user input: "
        f"'{user_text}'"
    )
    classification_result = classify_user_input(user_text)
    if classification_result == 1:
        print(
            f"[Callback:{callback_context.agent_name}] Classifier returned 1. "
            "Skipping agent's main logic."
        )
        predefined_response_text = prompt.JAILBREAK_FILTER_RESPONSE
        response_content = types.Content(
            role="model",
            parts=[types.Part(text=predefined_response_text)]
        )
        # Omit the violative event from being used in session memory
        callback_context.state["ignore_last_event"] = True
        return response_content
    else:
        print(
            f"[Callback:{callback_context.agent_name}] Classifier returned 0. "
            "Agent will process the input."
        )
        callback_context.state['ignore_last_event'] = False
        return None
