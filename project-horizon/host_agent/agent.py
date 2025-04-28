# host_agent/agent.py
import logging
import os
from google.adk.agents import Agent # Main ADK Agent class
from google.adk.agents.callback_context import CallbackContext # For callback type hints
from google.adk.tools import ToolContext, BaseTool # For callback type hints
from typing import Optional, Dict, Any # For callback type hints
from google.adk.agents.run_config import RunConfig


# --- Load Environment Variables ---
from dotenv import load_dotenv
# Assumes .env is in the parent directory (project root)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# --- Import the A2A Client Tool ---
try:
    # Use relative import assuming tools.py is in the same package
    from .tools import stock_a2a_tool
except ImportError:
    logging.critical("Failed to import 'stock_a2a_tool' from '.tools'. Ensure tools.py exists and A2A common library is accessible.")
    stock_a2a_tool = None # Set to None to prevent agent creation failure later

logger = logging.getLogger(__name__)

# --- Model Configuration ---
# IMPORTANT: Use a model ID that supports the Live API for run_live
# Check Google AI Studio or Vertex AI docs for current Live API compatible models
MODEL_ID_LIVE = os.getenv("GEMINI_LIVE_MODEL_ID", "gemini-2.0-flash-live-001") # Example, verify availability

# --- Optional Callback for Mocking ---
def check_mocking_before_tool(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """
    Checks 'mock_a2a_calls' state flag before calling the A2A tool.
    If True, returns a mock success response, skipping the actual A2A call.
    """
    # Only intercept the specific A2A tool
    if tool.name == "call_specialist_stock_agent":
        should_mock = tool_context.state.get("mock_a2a_calls", False)
        if should_mock:
            symbol = args.get("symbol", "MOCK")
            logger.warning(f"ADK Callback: MOCKING A2A tool call for symbol '{symbol}'.")
            # Return a dictionary matching the tool's success output structure
            return {"status": "success", "data": {"symbol": symbol.upper(), "price": 999.99, "currency": "USD", "mocked": True}}
        else:
            logger.info("ADK Callback: Mocking disabled. Allowing actual A2A tool call.")
            return None # Allow actual tool call
    else:
        # For any other tool, allow it to proceed
        return None

# --- Host Agent Definition ---
host_agent = None
if stock_a2a_tool: # Only define if the tool was imported successfully
    host_agent = Agent(
        name="HostAgentLive",
        model=MODEL_ID_LIVE, # Use the Live API compatible model ID
        description="User-facing agent handling voice/text and delegating stock queries via A2A.",
        instruction="You are a friendly financial assistant interacting via voice and text." # If the user says 'Hello' just say 'Hello' back.",
                    "Your primary function is to provide current stock prices. "
                    "When a user asks for the price of a specific stock (e.g., 'price of GOOGL', 'how is MSFT doing?'), "
                    "first identify the stock ticker symbol from their request. "
                    "Then, use the 'call_specialist_stock_agent' tool to get the price information. "
                    "This tool takes the 'symbol' as input. "
                    "Once you receive the result from the tool: "
                    "- If the status is 'success', clearly state the price and currency (e.g., 'Microsoft (MSFT) is currently trading at $150.25 USD.'). "
                    "- If the status is 'error', inform the user politely about the issue (e.g., 'Sorry, I couldn't retrieve the price for that symbol right now.'). "
                    "Handle other conversational turns naturally.",
        # Provide the A2A client tool
        tools=[stock_a2a_tool],
        # Add the callback to enable mocking via state
        # before_tool_callback=check_mocking_before_tool,
        # ADK's run_live implicitly handles multimodal input/output setup
        # when used with a compatible model and runner configuration.
    )
    logger.info(f"ADK Host Agent '{host_agent.name}' created with model '{MODEL_ID_LIVE}'.")
else:
    logger.critical("Host Agent could not be created because the A2A tool ('stock_a2a_tool') failed to load.")
    # Consider raising an exception or exiting if the host agent is critical
    # raise RuntimeError("Host Agent creation failed due to missing tool.")