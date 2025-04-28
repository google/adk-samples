# specialist_agents/stock_info_agent/__main__.py
import os
import logging
import click
from dotenv import load_dotenv

# A2A Imports from common library
from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill

# Local Imports
from .task_manager import StockInfoTaskManager

# --- Configuration ---
# Load .env file from the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - A2A_SERVER - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Function ---
@click.command()
@click.option("--host", default=os.getenv("SPECIALIST_SERVER_HOST", "127.0.0.1"), help="Host to bind the A2A server to.")
@click.option("--port", default=int(os.getenv("SPECIALIST_SERVER_PORT", "8001")), type=int, help="Port to bind the A2A server to.")
@click.option("--mcp-server-path",
              default=os.getenv("STOCK_MCP_SERVER_PATH"),
              help="Absolute path to the stock_mcp_server.py script.",
              required=not os.getenv("STOCK_MCP_SERVER_PATH") # Make required if not in env
             )
def main(host: str, port: int, mcp_server_path: str):
    """Starts the StockInfoAgent A2A Server."""
    logger.info("Starting StockInfoAgent A2A Server...")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  MCP Server Script Path: {mcp_server_path}")

    # --- Verify MCP Server Path ---
    if not os.path.isabs(mcp_server_path):
        logger.error(f"Error: Provided MCP server path is not absolute: '{mcp_server_path}'")
        click.echo("Error: MCP server path must be absolute.", err=True)
        return
    if not os.path.exists(mcp_server_path):
        logger.error(f"Error: MCP server script not found at: '{mcp_server_path}'")
        click.echo(f"Error: MCP server script not found at: '{mcp_server_path}'", err=True)
        return

    try:
        # --- Define Agent Card Programmatically ---
        capabilities = AgentCapabilities(streaming=False) # Synchronous agent
        skill = AgentSkill(
            id="get_stock_price_skill",
            name="Get Stock Price",
            description="Retrieves the current price and currency for a given stock ticker symbol.",
            tags=["finance", "stocks", "price lookup"],
            examples=["What is the price of GOOGL?", "Stock price for MSFT"],
            outputModes=["application/json"] # Skill specific output
        )
        agent_card = AgentCard(
            name="StockInfoAgent",
            description="Provides current stock price information using the yfinance library via MCP.",
            url=f"http://{host}:{port}/a2a", # Define the A2A endpoint path
            provider={"organization": "ProjectHorizon"},
            version="1.0.0",
            defaultInputModes=["text/plain"],
            defaultOutputModes=["application/json"],
            capabilities=capabilities,
            skills=[skill],
            authentication=None # No auth for this example
        )
        logger.info(f"Agent Card created for: {agent_card.name}")

        # --- Initialize Task Manager & A2A Server ---
        task_manager = StockInfoTaskManager(mcp_server_script_path=mcp_server_path)
        logger.debug("StockInfoTaskManager initialized.")

        # The A2AServer from common likely takes care of setting up ASGI routes
        # Pass host/port here if the common A2AServer uses them for its internal uvicorn run
        server = A2AServer(
            agent_card=agent_card,
            task_manager=task_manager,
            host=host,
            port=port,
            endpoint="/a2a" # Specify the endpoint path for A2A methods
        )
        logger.info(f"A2A Server configured to listen on {host}:{port} at endpoint '{server.endpoint}'")

        # --- Start the Server ---
        # server.start() likely calls uvicorn.run internally
        server.start()

    except ValueError as ve:
         logger.error(f"Configuration error during startup: {ve}")
         click.echo(f"Configuration Error: {ve}", err=True)
    except FileNotFoundError as fnf:
        logger.error(f"File not found error during startup: {fnf}")
        click.echo(f"File Not Found Error: {fnf}", err=True)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during server startup: {e}", exc_info=True)
        click.echo(f"Critical Error: {e}", err=True)

if __name__ == "__main__":
    main()