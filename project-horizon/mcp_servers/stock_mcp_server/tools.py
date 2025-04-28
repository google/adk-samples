import yfinance as yf
import logging

logger = logging.getLogger(__name__)

async def get_current_stock_price(ticker_symbol: str) -> dict:
    """
    Fetches the current stock price for a given ticker symbol.

    Args:
        ticker_symbol: The stock ticker symbol (e.g., 'GOOGL', 'MSFT').

    Returns:
        A dictionary with 'symbol', 'price', and 'currency',
        or an 'error' key if the lookup fails.
        Example success: {"symbol": "MSFT", "price": 410.50, "currency": "USD"}
        Example error: {"error": "Symbol 'XYZ' not found."}
    """
    logger.info(f"MCP Tool: Attempting to fetch price for {ticker_symbol}")
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Use 'currentPrice' for real-time, 'regularMarketPrice' often works too
        # 'info' can be slow; prefer direct price access if possible
        # price = ticker.info.get('currentPrice') or ticker.info.get('regularMarketPrice')
        # Faster alternative:
        hist = ticker.history(period="1d", interval="1m") # Get recent data
        if not hist.empty:
            price = hist['Close'].iloc[-1]
            currency = ticker.info.get('currency', 'USD') # Get currency if available
            logger.info(f"MCP Tool: Found price {price} {currency} for {ticker_symbol}")
            return {
                "symbol": ticker_symbol.upper(),
                "price": round(price, 2),
                "currency": currency
            }
        else:
            logger.warning(f"MCP Tool: No price data found for {ticker_symbol}")
            return {"error": f"Could not retrieve current price data for symbol '{ticker_symbol}'."}

    except Exception as e:
        logger.error(f"MCP Tool: Error fetching price for {ticker_symbol}: {e}")
        # Check if the error message suggests an invalid symbol
        if "No data found for symbol" in str(e) or "symbol may be delisted" in str(e):
             return {"error": f"Symbol '{ticker_symbol}' not found or data unavailable."}
        return {"error": f"An error occurred while fetching data for {ticker_symbol}: {str(e)}"}