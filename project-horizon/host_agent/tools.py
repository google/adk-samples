# adk_agents/host_agent/tools.py
import os
import uuid
import logging
import json
from typing import Dict, Any

# ADK Imports for ToolContext
from google.adk.tools import ToolContext, FunctionTool

# A2A Imports from common library
# Assuming 'common' is accessible from the project root
try:
    from common.client import A2AClient
    from common.types import (
        TaskSendParams, Message, TextPart, TaskState, DataPart,
        A2AClientHTTPError, A2AClientJSONError
    )
except ImportError:
    raise ImportError("Could not import A2A common library. Ensure 'common' directory is in the Python path or installed.")

logger = logging.getLogger(__name__)

# --- Tool Configuration ---
# Load the Specialist Agent URL from environment variables
SPECIALIST_A2A_URL = os.getenv("SPECIALIST_AGENT_A2A_URL")
if not SPECIALIST_A2A_URL:
    # Provide a default for local testing if not set, but log a warning
    logger.warning("SPECIALIST_AGENT_A2A_URL not set in environment, defaulting to http://127.0.0.1:8001/a2a")
    SPECIALIST_A2A_URL = "http://127.0.0.1:8001/a2a"

async def call_specialist_stock_agent(symbol: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Delegates a stock price lookup task to the specialized StockInfoAgent via A2A protocol.

    Args:
        symbol: The stock ticker symbol to look up (e.g., MSFT, GOOGL).
        tool_context: The ADK ToolContext, providing access to session info.

    Returns:
        A dictionary containing either the stock price data or an error message.
        Example success: {"status": "success", "data": {"symbol": "MSFT", "price": 410.50, "currency": "USD"}}
        Example error: {"status": "error", "message": "Could not get price for XYZ"}
    """
    logger.info(f"ADK Tool: call_specialist_stock_agent invoked for symbol: {symbol}")

    # Use the ADK session ID as the A2A session ID for correlation
    a2a_session_id = tool_context._invocation_context.session.id
    a2a_task_id = uuid.uuid4().hex
    logger.info(f"ADK Tool: Using A2A Session ID: {a2a_session_id}, Task ID: {a2a_task_id}")

    # Create A2A Client
    a2a_client = A2AClient(url=SPECIALIST_A2A_URL)

    # Prepare A2A Task
    task_params = TaskSendParams(
        id=a2a_task_id,
        sessionId=a2a_session_id,
        message=Message(
            role="user", # From the perspective of the specialist agent
            parts=[TextPart(text=symbol)]
        )
        # acceptedOutputModes=["application/json"] # Optional
    )

    try:
        logger.info(f"ADK Tool: Sending A2A task to {SPECIALIST_A2A_URL}...")
        # Log the details of the task being sent
        logger.info(f"ADK Tool: Sending TaskSendParams: {task_params.model_dump_json(indent=2)}")
        # Use .model_dump() for Pydantic v2+ included in common
        a2a_response = await a2a_client.send_task(task_params.model_dump())
        logger.info("ADK Tool: Received A2A response.")

        if a2a_response.error:
            error_msg = f"A2A protocol error: {a2a_response.error.message} (Code: {a2a_response.error.code})"
            logger.error(f"ADK Tool: {error_msg}")
            return {"status": "error", "message": error_msg}

        if a2a_response.result:
            task_result = a2a_response.result
            logger.debug(f"ADK Tool: A2A Task Result State: {task_result.status.state}")

            if task_result.status.state == TaskState.COMPLETED and task_result.artifacts:
                # Extract data from the first DataPart artifact
                for artifact in task_result.artifacts:
                     if artifact.name == "stock_price_data" and artifact.parts:
                         for part in artifact.parts:
                             if isinstance(part, DataPart):
                                 stock_data = part.data
                                 logger.info(f"ADK Tool: Successfully retrieved stock data: {stock_data}")
                                 return {"status": "success", "data": stock_data}
                # If artifact format wasn't as expected
                logger.warning("ADK Tool: Task completed but expected stock_price_data artifact not found or invalid.")
                return {"status": "error", "message": "Received unexpected completion format from specialist agent."}

            elif task_result.status.state == TaskState.FAILED and task_result.artifacts:
                 # Extract error from the error artifact
                 for artifact in task_result.artifacts:
                     if artifact.name == "error_details" and artifact.parts:
                          for part in artifact.parts:
                              if isinstance(part, DataPart) and "error" in part.data:
                                  error_msg = part.data["error"]
                                  logger.error(f"ADK Tool: Specialist agent task failed: {error_msg}")
                                  return {"status": "error", "message": f"Specialist Error: {error_msg}"}
                 # If error artifact format wasn't as expected
                 logger.error("ADK Tool: Specialist agent task failed, but couldn't parse error artifact.")
                 return {"status": "error", "message": "Specialist agent reported failure with unclear details."}
            else:
                 # Handle other unexpected states
                 logger.error(f"ADK Tool: Specialist agent task ended in unexpected state: {task_result.status.state}")
                 return {"status": "error", "message": f"Specialist agent ended in state: {task_result.status.state}"}
        else:
            logger.error("ADK Tool: Received empty successful response from A2A server.")
            return {"status": "error", "message": "Empty response from specialist agent."}

    except A2AClientHTTPError as http_err:
         logger.error(f"ADK Tool: HTTP Error calling specialist agent: {http_err.status_code} - {http_err.message}", exc_info=True)
         return {"status": "error", "message": f"Network error communicating with specialist: {http_err.status_code}"}
    except A2AClientJSONError as json_err:
         logger.error(f"ADK Tool: JSON Error processing specialist response: {json_err.message}", exc_info=True)
         return {"status": "error", "message": "Invalid response format from specialist."}
    except ConnectionRefusedError:
         logger.error(f"ADK Tool: Connection refused. Is the specialist A2A server running at {SPECIALIST_A2A_URL}?", exc_info=True)
         return {"status": "error", "message": "Could not connect to specialist agent."}
    except Exception as e:
        logger.error(f"ADK Tool: Unexpected error during A2A call: {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

# Wrap the async function in an ADK FunctionTool
stock_a2a_tool = FunctionTool(
    func=call_specialist_stock_agent,
    # The name here MUST match the function name if not provided explicitly
    # name="call_specialist_stock_agent",
    # description is taken from the docstring
)

print("✅ ADK Tool for A2A communication defined.")