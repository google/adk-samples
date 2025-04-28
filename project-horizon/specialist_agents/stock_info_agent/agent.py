# live-agent-project/adk_agents/stock_info_agent/agent.py

import asyncio
import os
import sys
import logging
from contextlib import AsyncExitStack # Crucial for managing the MCP connection

# --- ADK Imports ---
from google.adk.agents import Agent  # Using the alias
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types # Use alias to avoid confusion

# --- MCP Client Imports ---
# Import the toolset and connection parameters from ADK's MCP integration
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# --- Environment Loading (Optional but recommended) ---
from dotenv import load_dotenv
# Load .env file from the parent directory (live-agent-project)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
print(f"dotenv_path: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path, override=True)



# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In specialist_agents/stock_info_agent/agent.py, after load_dotenv:
logger.info(f"--- Loaded Environment Variables ---")
logger.info(f"VERTEX_AI?: {os.getenv('GOOGLE_GENAI_USE_VERTEXAI')}")
logger.info(f"PROJECT_ID: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
logger.info(f"LOCATION:   {os.getenv('GOOGLE_CLOUD_LOCATION')}")
logger.info(f"---------------------------------")

# --- Constants ---
APP_NAME = "stock_info_adk_app"
USER_ID = "test_user_stock"
SESSION_ID = "session_stock_001"
MODEL = "gemini-2.0-flash-001" # Ensure this model is available via your GOOGLE_API_KEY

# --- Dynamically find the MCP Server script path ---
# This assumes the adk_agent script is in adk_agents/stock_info_agent/
# and the mcp server script is in mcp_servers/stock_mcp_server/
# relative to the project root.
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    MCP_SERVER_SCRIPT_NAME = "server.py"
    MCP_SERVER_FOLDER = os.path.join(PROJECT_ROOT, "mcp_servers", "stock_mcp_server")
    MCP_SERVER_SCRIPT_PATH = os.path.join(MCP_SERVER_FOLDER, MCP_SERVER_SCRIPT_NAME)

    # Verify the path exists
    if not os.path.isfile(MCP_SERVER_SCRIPT_PATH):
        raise FileNotFoundError(f"MCP Server script not found at calculated path: {MCP_SERVER_SCRIPT_PATH}")
    logger.info(f"Found MCP server script path: {MCP_SERVER_SCRIPT_PATH}")

except Exception as e:
    logger.critical(f"CRITICAL ERROR: Could not determine MCP server path. {e}")
    sys.exit(1) # Exit if the path is wrong, as the agent cannot function

# --- MCP Tool Loading Function ---
async def get_mcp_tools_async(mcp_server_script_path: str) -> tuple[list, AsyncExitStack]:
    """
    Connects to the MCP server via stdio and retrieves its tools.

    Args:
        mcp_server_script_path: The absolute path to the MCP server Python script.

    Returns:
        A tuple containing the list of ADK-compatible tools and the AsyncExitStack
        for managing the connection lifecycle.
    """
    logger.info(f"Attempting to connect to MCP server via stdio: {mcp_server_script_path}")
    try:
        # Configure parameters to launch the MCP server script via stdio
        connection_params = StdioServerParameters(
            command=sys.executable,  # Use the current Python interpreter
            args=[mcp_server_script_path], # Pass the script path as an argument
            # Add cwd if necessary, depending on how server.py resolves things
            # cwd=os.path.dirname(mcp_server_script_path)
        )

        # Use ADK's MCPToolset to connect and get tools
        # This starts the server.py script as a subprocess
        tools, exit_stack = await MCPToolset.from_server(
            connection_params=connection_params
        )
        logger.info(f"Successfully connected to MCP server and retrieved {len(tools)} tool(s).")
        return tools, exit_stack
    except Exception as e:
        logger.error(f"Failed to connect to MCP server or get tools: {e}", exc_info=True)
        raise # Re-raise the exception to stop execution if connection fails

# --- ADK Agent Definition Function ---
async def create_agent_with_mcp_tools(mcp_server_script_path: str) -> tuple[Agent, AsyncExitStack]:
    """
    Creates the ADK Agent, equipping it with tools loaded from the MCP server.

    Args:
        mcp_server_script_path: Absolute path to the MCP server script.

    Returns:
        A tuple containing the configured ADK Agent instance and the AsyncExitStack.
    """
    mcp_tools, exit_stack = await get_mcp_tools_async(mcp_server_script_path)

    # Define the ADK Agent that will use the MCP tool(s)
    stock_info_agent = Agent(
        name="stock_info_agent",
        model=MODEL,
        description="Provides current stock price information using an external tool.",
        instruction="You are an assistant that provides stock prices. "
                    "When asked for the price of a stock, use the 'get_current_stock_price' tool. "
                    "The tool takes a 'symbol' argument (e.g., 'MSFT'). "
                    "The tool returns a dictionary with 'price' and 'currency', or an 'error'. "
                    "Relay the information clearly to the user. If the tool returns an error, state that.",
        # Provide the tools loaded from the MCP server to the ADK agent
        tools=mcp_tools,
    )
    logger.info(f"ADK Agent '{stock_info_agent.name}' created with MCP tools.")
    return stock_info_agent, exit_stack

# --- Main Execution Logic ---
async def async_main():
    """Sets up the runner and runs a sample query against the agent."""
    agent = None
    exit_stack = None
    try:
        # Create the agent and get the exit stack for cleanup
        agent, exit_stack = await create_agent_with_mcp_tools(MCP_SERVER_SCRIPT_PATH)

        # Standard ADK setup
        session_service = InMemorySessionService()
        runner = Runner(
            agent=agent,
            app_name=APP_NAME,
            session_service=session_service,
            # artifact_service=... # Add if needed
            # memory_service=... # Add if needed
        )
        logger.info("ADK Runner initialized.")

        # Create a session
        session = session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID, state={}
        )
        logger.info(f"Session '{SESSION_ID}' created.")

        # Define the user query
        user_query = "What is the current price of Microsoft stock (MSFT)?"
        logger.info(f"User Query: '{user_query}'")
        content = genai_types.Content(role='user', parts=[genai_types.Part(text=user_query)])

        # Run the agent
        logger.info("Running agent execution...")
        events_async = runner.run_async(
            session_id=session.id, user_id=session.user_id, new_message=content
        )

        final_response_text = "Agent did not produce a final response."
        async for event in events_async:
            logger.info(f"Received Event: Author={event.author}, Content Present={bool(event.content)}, Actions Present={bool(event.actions)}")
            # Optional: More detailed logging of event parts
            # if event.content:
            #     logger.debug(f"  Event Content Parts: {event.content.parts}")
            if event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text or "[Non-text response]"

        logger.info(f"Agent Final Response: {final_response_text}")
        logger.info("Agent execution finished.")

    except FileNotFoundError as fnf_error:
         logger.critical(f"Setup failed: {fnf_error}. Cannot proceed.")
    except Exception as e:
        logger.error(f"An error occurred during agent execution: {e}", exc_info=True)
    finally:
        # CRUCIAL: Ensure the MCP connection is closed
        if exit_stack:
            logger.info("Closing MCP server connection via exit stack...")
            await exit_stack.aclose()
            logger.info("MCP server connection closed.")
        else:
             logger.warning("Exit stack was not initialized, cannot close MCP connection cleanly.")

# --- Script Entry Point ---
if __name__ == "__main__":
    logger.info("Starting ADK agent script...")
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
    finally:
        logger.info("ADK agent script finished.")