#!/usr/bin/env python3
"""
Arize-based evaluation suite for the Travel Concierge agent.
This follows the Arize documentation pattern for experiments while using Google Vertex AI for evaluations.
"""

import json
import os
import time
import uuid
from typing import Any

import pandas as pd

# Google Cloud imports for Vertex AI evaluations
import vertexai

# Arize imports
from arize.experimental.datasets import ArizeDatasetsClient
from arize.experimental.datasets.experiments.types import EvaluationResult
from arize.experimental.datasets.utils.constants import GENERATIVE
from arize_eval_templates import (
    AGENT_HANDOFF_TEMPLATE,
    RESPONSE_QUALITY_TEMPLATE,
    TOOL_USAGE_TEMPLATE,
)
from dotenv import load_dotenv

# ADK imports for running the agent
from google.adk.runners import InMemoryRunner
from google.genai.types import Part, UserContent

# Phoenix Evals imports
from phoenix.evals import GeminiModel, llm_classify

# Import the travel concierge agent
from travel_concierge import MODEL
from travel_concierge.agent import root_agent

load_dotenv()

# Environment variables
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

if not all([ARIZE_API_KEY, ARIZE_SPACE_ID, GOOGLE_CLOUD_PROJECT]):
    raise ValueError(
        "Missing required environment variables: ARIZE_API_KEY, ARIZE_SPACE_ID, GOOGLE_CLOUD_PROJECT"
    )

# Initialize Vertex AI
vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)

# Initialize Arize client
arize_client = ArizeDatasetsClient(api_key=ARIZE_API_KEY)

# Initialize Phoenix model for evaluations
phoenix_model = GeminiModel(
    model=MODEL,
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
)


def load_test_data(filename: str) -> dict[str, Any]:
    """Load the conversation test data from JSON file."""
    with open(f"eval/data/{filename}") as f:
        return json.load(f)


def extract_conversation_data(test_data: dict[str, Any]) -> list[dict]:
    """Extract individual queries and expected tool usage from conversation data."""
    dataset_rows = []

    for eval_case in test_data.get("eval_cases", []):
        conversation = eval_case.get("conversation", [])
        session_input = eval_case.get("session_input", {})

        for i, turn in enumerate(conversation):
            user_content = turn.get("user_content", {})
            final_response = turn.get("final_response", {})
            intermediate_data = turn.get("intermediate_data", {})

            # Extract user query
            user_parts = user_content.get("parts", [])
            query = ""
            for part in user_parts:
                if part.get("text"):
                    query = part["text"]
                    break

            # Extract final response
            response_parts = final_response.get("parts", [])
            response = ""
            for part in response_parts:
                if part.get("text"):
                    response = part["text"]
                    break

            # Extract tool uses (these become our "expected" tool usage)
            tool_uses = intermediate_data.get("tool_uses", [])

            # Extract agent transfers
            agent_transfers = []
            other_tools = []
            for tool_use in tool_uses:
                if tool_use.get("name") == "transfer_to_agent":
                    agent_transfers.append(
                        tool_use.get("args", {}).get("agent_name", "")
                    )
                else:
                    other_tools.append(tool_use)

            if query:  # Only add if there's a user query
                dataset_rows.append(
                    {
                        "id": f"{eval_case.get('eval_id', 'unknown')}_{i}",
                        "query": query,
                        "expected_response": response,
                        "expected_tool_uses": json.dumps(tool_uses),
                        "expected_agent_transfers": json.dumps(agent_transfers),
                        "expected_other_tools": json.dumps(other_tools),
                        "session_state": json.dumps(
                            session_input.get("state", {})
                        ),
                        "conversation_context": (
                            json.dumps(conversation[:i]) if i > 0 else "[]"
                        ),
                    }
                )

    return dataset_rows


def create_arize_dataset(dataset_name: str, filename: str) -> dict[str, str]:
    """Create an Arize dataset from the test data."""
    test_data = load_test_data(filename)
    dataset_rows = extract_conversation_data(test_data)

    # Create DataFrame
    df = pd.DataFrame(dataset_rows)

    # Debug: Print the DataFrame structure
    print(f"Dataset {dataset_name} - DataFrame shape:", df.shape)
    print(f"Dataset {dataset_name} - DataFrame columns:", df.columns.tolist())
    print(
        f"Dataset {dataset_name} - First row sample:",
        df.iloc[0].to_dict() if len(df) > 0 else "No rows",
    )

    # Create dataset in Arize
    full_dataset_name = (
        f"travel_concierge_{dataset_name}_evaluation_dataset-{uuid.uuid4()}"
    )

    print(f"Creating dataset: {full_dataset_name}")
    dataset_id = arize_client.create_dataset(
        space_id=ARIZE_SPACE_ID,
        dataset_name=full_dataset_name,
        data=df,
        dataset_type=GENERATIVE,
    )

    print(f"Dataset created with ID: {dataset_id}")
    time.sleep(5)  # Wait after dataset creation
    return {"id": dataset_id, "name": full_dataset_name}


def extract_tool_calls_from_response(
    response_text: str, agent_runner
) -> dict[str, Any]:
    """
    Extract tool call information from the agent's response and execution context.
    This creates metadata for evaluation.
    """
    # For the travel concierge, we'll need to examine execution traces
    # This is a simplified implementation - in practice, we'd need to capture
    # actual tool calls from the ADK agent execution

    max_response_length = 100

    tool_calls = []
    agent_transfers = []

    # Simple heuristics to detect potential tool usage
    if "transfer" in response_text.lower():
        # Look for agent transfer patterns
        if "inspiration" in response_text.lower():
            agent_transfers.append("inspiration_agent")
        elif "planning" in response_text.lower():
            agent_transfers.append("planning_agent")
        elif "booking" in response_text.lower():
            agent_transfers.append("booking_agent")
        elif "pre" in response_text.lower() and "trip" in response_text.lower():
            agent_transfers.append("pre_trip_agent")
        elif "in" in response_text.lower() and "trip" in response_text.lower():
            agent_transfers.append("in_trip_agent")
        elif (
            "post" in response_text.lower() and "trip" in response_text.lower()
        ):
            agent_transfers.append("post_trip_agent")

    # Detect other tool usage patterns
    if any(
        keyword in response_text.lower()
        for keyword in ["search", "flight", "hotel", "weather", "availability"]
    ):
        tool_calls.append(
            {
                "tool_name": "search_tool",  # Generic inference
                "tool_input": (
                    response_text[:max_response_length] + "..."
                    if len(response_text) > max_response_length
                    else response_text
                ),
            }
        )

    return {"tool_calls": tool_calls, "agent_transfers": agent_transfers}


async def call_travel_concierge_agent(
    query: str,
    session_state: dict | None = None,
    conversation_context: list | None = None,
) -> dict[str, Any]:
    """Call the travel concierge agent programmatically and return response with metadata."""
    runner = InMemoryRunner(agent=root_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="test_user"
    )

    # TODO: Set session state if provided
    # This would require understanding how to set the travel concierge session state

    content = UserContent(parts=[Part(text=query)])
    response_parts = []

    for event in runner.run(
        user_id=session.user_id, session_id=session.id, new_message=content
    ):
        if event and event.content and event.content.parts:
            for part in event.content.parts:
                if part and hasattr(part, "text") and part.text:
                    response_parts.append(part.text)

    response_text = "\n".join(response_parts)

    # Extract tool usage information
    tool_metadata = extract_tool_calls_from_response(response_text, runner)

    return {
        "response": response_text,
        "tool_calls": tool_metadata["tool_calls"],
        "agent_transfers": tool_metadata["agent_transfers"],
    }


async def task_function(dataset_row: dict) -> str:
    """
    Task function for Arize experiments.
    This calls the travel concierge agent and returns the response.
    """
    query = dataset_row.get("query", "")
    session_state = json.loads(dataset_row.get("session_state", "{}"))
    conversation_context = json.loads(
        dataset_row.get("conversation_context", "[]")
    )

    # Call the agent
    result = await call_travel_concierge_agent(
        query, session_state, conversation_context
    )

    # Store metadata for evaluations
    metadata = {
        "agent_response": result["response"],
        "actual_tool_calls": result["tool_calls"],
        "actual_agent_transfers": result["agent_transfers"],
        "expected_tool_uses": dataset_row.get("expected_tool_uses", "[]"),
        "expected_agent_transfers": dataset_row.get(
            "expected_agent_transfers", "[]"
        ),
        "expected_other_tools": dataset_row.get("expected_other_tools", "[]"),
        "expected_response": dataset_row.get("expected_response", ""),
    }

    return json.dumps(metadata)


def agent_handoff_evaluator(output: str, dataset_row: dict) -> EvaluationResult:
    """Evaluator for agent handoff correctness."""
    try:
        # Parse the agent output to get the metadata
        try:
            metadata = json.loads(output) if isinstance(output, str) else output
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        # Create evaluation data for this specific row
        eval_data = pd.DataFrame(
            [
                {
                    "query": dataset_row.get("query", ""),
                    "agent_response": metadata.get("agent_response", ""),
                    "expected_agent_transfers": dataset_row.get(
                        "expected_agent_transfers", "[]"
                    ),
                    "actual_agent_transfers": json.dumps(
                        metadata.get("actual_agent_transfers", [])
                    ),
                }
            ]
        )

        # Run the Phoenix evaluation for this specific row
        handoff_result = llm_classify(
            data=eval_data,
            model=phoenix_model,
            template=AGENT_HANDOFF_TEMPLATE,
            rails=AGENT_HANDOFF_TEMPLATE.rails,
            verbose=False,
        )

        # Extract the result
        if len(handoff_result) > 0 and "label" in handoff_result.columns:
            label = handoff_result.iloc[0]["label"]
        else:
            label = "unknown"

        # Binary scoring: 1.0 for correct, 0.0 for incorrect
        score = 1.0 if label == "correct_handoff" else 0.0

        explanation = f"Agent handoff evaluation: {label}. Expected transfers: {dataset_row.get('expected_agent_transfers', 'N/A')}, Actual transfers: {json.dumps(metadata.get('actual_agent_transfers', []))}"

        return EvaluationResult(
            score=score, label=label, explanation=explanation
        )

    except Exception as e:
        print(f"Error in agent_handoff_evaluator: {e}")
        return EvaluationResult(
            score=0.0,
            label="error",
            explanation=f"Error during evaluation: {e!s}",
        )


def tool_usage_evaluator(output: str, dataset_row: dict) -> EvaluationResult:
    """Evaluator for tool usage correctness."""
    try:
        # Parse the agent output to get the metadata
        try:
            metadata = json.loads(output) if isinstance(output, str) else output
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        # Create evaluation data for this specific row
        eval_data = pd.DataFrame(
            [
                {
                    "query": dataset_row.get("query", ""),
                    "agent_response": metadata.get("agent_response", ""),
                    "expected_other_tools": dataset_row.get(
                        "expected_other_tools", "[]"
                    ),
                    "actual_tool_calls": json.dumps(
                        metadata.get("actual_tool_calls", [])
                    ),
                }
            ]
        )

        # Run the Phoenix evaluation for this specific row
        tool_result = llm_classify(
            data=eval_data,
            model=phoenix_model,
            template=TOOL_USAGE_TEMPLATE,
            rails=TOOL_USAGE_TEMPLATE.rails,
            verbose=False,
        )

        # Extract the result
        if len(tool_result) > 0 and "label" in tool_result.columns:
            label = tool_result.iloc[0]["label"]
        else:
            label = "unknown"

        # Binary scoring: 1.0 for correct, 0.0 for incorrect
        score = 1.0 if label == "correct_tools" else 0.0

        explanation = f"Tool usage evaluation: {label}. Expected tools: {dataset_row.get('expected_other_tools', 'N/A')}, Actual tools: {json.dumps(metadata.get('actual_tool_calls', []))}"

        return EvaluationResult(
            score=score, label=label, explanation=explanation
        )

    except Exception as e:
        print(f"Error in tool_usage_evaluator: {e}")
        return EvaluationResult(
            score=0.0,
            label="error",
            explanation=f"Error during evaluation: {e!s}",
        )


def response_quality_evaluator(
    output: str, dataset_row: dict
) -> EvaluationResult:
    """Evaluator for response quality."""
    try:
        # Parse the agent output to get the metadata
        try:
            metadata = json.loads(output) if isinstance(output, str) else output
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        # Create evaluation data for this specific row
        eval_data = pd.DataFrame(
            [
                {
                    "query": dataset_row.get("query", ""),
                    "agent_response": metadata.get("agent_response", ""),
                    "expected_response": dataset_row.get(
                        "expected_response", ""
                    ),
                }
            ]
        )

        # Run the Phoenix evaluation for this specific row
        quality_result = llm_classify(
            data=eval_data,
            model=phoenix_model,
            template=RESPONSE_QUALITY_TEMPLATE,
            rails=RESPONSE_QUALITY_TEMPLATE.rails,
            verbose=False,
        )

        # Extract the result
        if len(quality_result) > 0 and "label" in quality_result.columns:
            label = quality_result.iloc[0]["label"]
        else:
            label = "unknown"

        # Binary scoring: 1.0 for good, 0.0 for poor
        score = 1.0 if label == "good_response" else 0.0

        explanation = f"Response quality evaluation: {label}. Query: {dataset_row.get('query', 'N/A')[:100]}..."

        return EvaluationResult(
            score=score, label=label, explanation=explanation
        )

    except Exception as e:
        print(f"Error in response_quality_evaluator: {e}")
        return EvaluationResult(
            score=0.0,
            label="error",
            explanation=f"Error during evaluation: {e!s}",
        )


def run_evaluation_experiment():
    """Run the complete evaluation experiment using Arize with Phoenix evaluations."""

    # Create datasets for each test file
    datasets = []
    test_files = [
        ("inspire", "inspire.test.json"),
        ("pretrip", "pretrip.test.json"),
        ("intrip", "intrip.test.json"),
    ]

    for dataset_name, filename in test_files:
        print(f"Creating Arize dataset for {dataset_name}...")
        dataset = create_arize_dataset(dataset_name, filename)
        datasets.append((dataset_name, dataset))

    # Run experiments for each dataset
    experiment_results = []

    for dataset_name, dataset in datasets:
        print(f"Running agent tasks for {dataset_name}...")

        # First, run the agent tasks to get responses
        # We'll collect the results manually for Phoenix evaluation
        # Use the 3 separate named evaluator functions
        evaluators = [
            agent_handoff_evaluator,
            tool_usage_evaluator,
            response_quality_evaluator,
        ]

        print(f"Running Arize experiment for {dataset_name}...")

        # Run Arize experiment with Phoenix evaluators
        experiment_result = arize_client.run_experiment(
            space_id=ARIZE_SPACE_ID,
            dataset_id=dataset["id"],
            task=task_function,
            evaluators=evaluators,
            experiment_name=f"travel_concierge_phoenix_{dataset_name}_evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            concurrency=2,  # Reduce concurrency to be gentle on APIs
            exit_on_error=False,
            dry_run=False,
        )

        # Handle the experiment result
        if hasattr(experiment_result, "id"):
            experiment_id = experiment_result.id
        elif (
            isinstance(experiment_result, tuple) and len(experiment_result) > 0
        ):
            experiment_id = (
                experiment_result[0].id
                if hasattr(experiment_result[0], "id")
                else str(experiment_result[0])
            )
        else:
            experiment_id = "unknown"

        experiment_results.append((dataset_name, experiment_id))
        print(
            f"Experiment for {dataset_name} completed! Experiment ID: {experiment_id}"
        )

    print("\n=== All Experiments Completed ===")
    for dataset_name, experiment_id in experiment_results:
        print(f"{dataset_name}: {experiment_id}")
    print("View results in the Arize UI")

    return experiment_results


if __name__ == "__main__":
    run_evaluation_experiment()
