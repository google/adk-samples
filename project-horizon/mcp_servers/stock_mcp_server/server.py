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


# @mcp.tool()
# async def get_stock_price(symbol: str) -> dict:
#     """
#     Retrieve the current stock price for the given ticker symbol.
#     Returns a dictionary containing the price and currency, or an error message.
#     Example success: {"symbol": "MSFT", "price": 410.50, "currency": "USD"}
#     Example error: {"error": "Symbol 'XYZ' not found."}
#     """
#     logger.info(f"Tool: Attempting to fetch price for {symbol}")
#     try:
#         ticker = yf.Ticker(symbol)
#         # Using history might be more reliable for current price than info['regularMarketPrice']
#         hist = ticker.history(period="1d", interval="1m") # Get recent data
#         if not hist.empty:
#             price = hist['Close'].iloc[-1]
#             currency = ticker.info.get('currency', 'USD')
#             result = {
#                 "symbol": symbol.upper(),
#                 "price": round(price, 2),
#                 "currency": currency
#             }
#             logger.info(f"Tool: Found price {result['price']} {result['currency']} for {symbol}")
#             return result
#         else:
#             # Fallback if 1d history is empty (e.g., non-market hours, less common ticker)
#             info = ticker.info
#             price = info.get("regularMarketPrice") or info.get("currentPrice")
#             if price is not None:
#                  currency = info.get('currency', 'USD')
#                  result = {
#                      "symbol": symbol.upper(),
#                      "price": round(float(price), 2),
#                      "currency": currency
#                  }
#                  logger.info(f"Tool: Found price (fallback) {result['price']} {result['currency']} for {symbol}")
#                  return result
#             else:
#                 logger.warning(f"Tool: No price data found for {symbol}")
#                 return {"error": f"Could not retrieve current price data for symbol '{symbol}'."}
#     except Exception as e:
#         logger.error(f"Tool: Error fetching price for {symbol}: {e}")
#         # Check if the error message suggests an invalid symbol
#         if "No data found for symbol" in str(e) or "symbol may be delisted" in str(e) or "Failed to get ticker" in str(e):
#              return {"error": f"Symbol '{symbol}' not found or data unavailable."}
#         return {"error": f"An error occurred while fetching data for {symbol}: {str(e)}"}


# # --- Example Resource (Optional, keep if needed) ---
# @mcp.resource("stock://{symbol}")
# async def stock_resource(symbol: str) -> str:
#     """
#     Expose stock price data as a resource.
#     Returns a formatted string with the current stock price.
#     """
#     logger.info(f"Resource: Request for stock://{symbol}")
#     price_data = await get_stock_price(symbol) # Reuse the tool logic
#     if "error" in price_data:
#         return f"Error: {price_data['error']}"
#     return f"The current price of '{symbol.upper()}' is ${price_data['price']:.2f} {price_data['currency']}."

# # --- Example History Tool (Optional, keep if needed) ---
# @mcp.tool()
# async def get_stock_history(symbol: str, period: str = "1mo") -> dict:
#     """
#     Retrieve historical data for a stock.
#     Returns a dictionary containing CSV data or an error message.

#     Args:
#         symbol: The stock ticker symbol.
#         period: The period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', 'ytd', 'max'). Default '1mo'.
#     """
#     logger.info(f"Tool: Attempting to fetch history for {symbol}, period {period}")
#     try:
#         ticker = yf.Ticker(symbol)
#         data = ticker.history(period=period)
#         if data.empty:
#             logger.warning(f"Tool: No history found for {symbol} ({period})")
#             return {"error": f"No historical data found for symbol '{symbol}' with period '{period}'."}

#         csv_data = data.to_csv()
#         logger.info(f"Tool: Successfully fetched history for {symbol} ({period})")
#         return {"csv_data": csv_data} # Return as structured data

#     except Exception as e:
#         logger.error(f"Tool: Error fetching history for {symbol}: {e}")
#         return {"error": f"Error fetching historical data: {str(e)}"}


# # --- Example Comparison Tool (Optional, keep if needed) ---
# @mcp.tool()
# async def compare_stocks(symbol1: str, symbol2: str) -> dict:
#     """
#     Compare the current stock prices of two ticker symbols.
#     Returns a dictionary containing a comparison message or an error.
#     """
#     logger.info(f"Tool: Comparing stocks {symbol1} and {symbol2}")
#     price1_data = await get_stock_price(symbol1)
#     price2_data = await get_stock_price(symbol2)

#     if "error" in price1_data or "error" in price2_data:
#         err1 = price1_data.get("error", "N/A")
#         err2 = price2_data.get("error", "N/A")
#         logger.warning(f"Tool: Error comparing stocks. Price1 Error: {err1}, Price2 Error: {err2}")
#         return {"error": f"Could not retrieve data for comparison. Error for {symbol1}: {err1}. Error for {symbol2}: {err2}."}

#     price1 = price1_data["price"]
#     price2 = price2_data["price"]
#     curr1 = price1_data["currency"]
#     curr2 = price2_data["currency"]

#     # Basic comparison, assumes same currency for simplicity
#     if price1 > price2:
#         result = f"{symbol1.upper()} (${price1:.2f} {curr1}) is higher than {symbol2.upper()} (${price2:.2f} {curr2})."
#     elif price1 < price2:
#         result = f"{symbol1.upper()} (${price1:.2f} {curr1}) is lower than {symbol2.upper()} (${price2:.2f} {curr2})."
#     else:
#         result = f"Both {symbol1.upper()} and {symbol2.upper()} have the same price (${price1:.2f} {curr1})."

#     logger.info(f"Tool: Comparison result: {result}")
#     return {"comparison": result}

# @mcp.tool()
# async def echo(message: str) -> dict:
#     """A simple tool that echoes back the received message."""
#     logger.info(f"Tool: echo called with message: '{message}'")
#     # No blocking calls, should return immediately
#     return {"result": {"received": message}}


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting FastMCP Stock Price Server...")
    # mcp.run() defaults to stdio transport
    # It handles the initialization handshake automatically
    mcp.run()
    logger.info("FastMCP Stock Price Server stopped.")