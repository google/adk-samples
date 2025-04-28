# live-agent-project/mcp_servers/stock_mcp_server/server.py
import logging
import asyncio
from mcp.server.fastmcp import FastMCP
import yfinance as yf
import sys
import json
# --- Configuration ---
# Basic logging setup
logging.basicConfig(level=logging.DEBUG, # <--- Change level to DEBUG for more detail
                    stream=sys.stderr,
                    format='%(asctime)s - %(name)s - %(levelname)s - MCP_SERVER - %(message)s')
logger = logging.getLogger(__name__)

# --- FastMCP Server Initialization ---
# Creates a customizable MCP server named "Stock Price Server"
logger.info("Initializing FastMCP server...")
mcp = FastMCP("Stock Price Server") # Name used during initialization

# --- Tool Functions (using decorators) ---

# Note: The get_stock_price function is now directly decorated as a tool.
# It's generally better practice for tools to return structured data (like dicts)
# rather than just floats or strings, to allow for error reporting.

@mcp.tool()
async def get_current_stock_price(symbol: str) -> dict:
    """
    Retrieve the current stock price for the given ticker symbol.
    Returns a dictionary containing the price and currency, or an error message.
    """
    # Log the incoming arguments (part of the MCP request's params)
    logger.debug(f"Tool 'get_current_stock_price' called with args: {{'symbol': '{symbol}'}}")
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="1m")

        if not hist.empty:
            price = hist['Close'].iloc[-1]
            currency = ticker.info.get('currency', 'USD')
            result = {
                "symbol": symbol.upper(),
                "price": round(price, 2),
                "currency": currency
            }
            logger.info(f"Tool: Successfully found price {result['price']} {result['currency']} for {symbol} using history.")
        else:
            info = ticker.info
            price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
            if price is not None:
                 currency = info.get('currency', 'USD')
                 result = {
                     "symbol": symbol.upper(),
                     "price": round(float(price), 2),
                     "currency": currency
                 }
                 logger.info(f"Tool: Successfully found price {result['price']} {result['currency']} for {symbol} using fallback info.")
            else:
                logger.warning(f"Tool: No current price data found for symbol {symbol} via history or info.")
                result = {"error": f"Could not retrieve current price data for symbol '{symbol}'."}

    except Exception as e:
        logger.error(f"Tool: Error fetching price for {symbol}: {e}", exc_info=False) # Set exc_info=False if stack trace is too verbose
        if "No data found for symbol" in str(e) or "symbol may be delisted" in str(e) or "Failed to get ticker" in str(e):
             result = {"error": f"Symbol '{symbol}' not found or data unavailable."}
        else:
             result = {"error": f"An error occurred while fetching data for {symbol}."}

    # Log the result dictionary (which will be the MCP response's result field)
    logger.debug(f"Tool 'get_current_stock_price' returning result: {json.dumps(result, indent=2)}")
    return result


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting FastMCP Stock Price Server...")
    # mcp.run() defaults to stdio transport
    # It handles the initialization handshake automatically
    mcp.run()
    logger.info("FastMCP Stock Price Server stopped.")