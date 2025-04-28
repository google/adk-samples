# specialist_agents/stock_info_agent/mcp_client_logic.py
import asyncio
import sys
import os
import logging
import json # <--- Import json
from contextlib import AsyncExitStack

# MCP Client Imports
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
# Import content types to check against
from mcp.types import CallToolResult, TextContent  #, DataPart # <-- Import relevant types

logger = logging.getLogger(__name__)

async def call_mcp_get_stock_price(symbol: str, mcp_server_script_path: str) -> dict:
    """
    Launches the MCP server via stdio, calls the 'get_current_stock_price' tool,
    and returns the result dictionary. Handles JSON parsing from TextContent.
    """
    logger.info(f"MCP Client Logic: Launching MCP server: {mcp_server_script_path} for symbol {symbol}")

    if not os.path.isabs(mcp_server_script_path):
        logger.error(f"MCP Client Logic: Path '{mcp_server_script_path}' is not absolute.")
        return {"error": "Internal configuration error: MCP server path is not absolute."}
    if not os.path.exists(mcp_server_script_path):
         logger.error(f"MCP Client Logic: MCP server script not found at: {mcp_server_script_path}")
         return {"error": f"Internal configuration error: MCP server script not found."}

    params = StdioServerParameters(
        command=sys.executable,
        args=[mcp_server_script_path],
    )

    async with AsyncExitStack() as stack:
        session: ClientSession = None
        try:
            streams = await stack.enter_async_context(stdio_client(params))
            logger.debug("MCP Client Logic: stdio streams acquired.")

            session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
            logger.debug("MCP Client Logic: ClientSession established.")

            await session.initialize()
            logger.info("MCP Client Logic: Session initialized with MCP server.")

            tool_name = "get_current_stock_price"
            arguments = {"symbol": symbol}
            logger.info(f"MCP Client Logic: Calling tool '{tool_name}' with args: {arguments}")

            mcp_response: CallToolResult = await session.call_tool(tool_name, arguments)
            logger.info("MCP Client Logic: Received response from MCP server.")

            # --- Process the MCP Response (REVISED LOGIC) ---
            if mcp_response.isError:
                logger.warning("MCP Client Logic: MCP tool call reported an error.")
                error_content = "Unknown tool error"
                if mcp_response.content:
                    try:
                        # Check if it's TextContent and extract text
                        if isinstance(mcp_response.content[0], TextContent):
                             error_content = mcp_response.content[0].text
                        # Add checks for other content types if necessary
                        else:
                             error_content = str(mcp_response.content[0]) # Fallback
                    except (IndexError, AttributeError, TypeError):
                        error_content = str(mcp_response.content)
                return {"error": f"MCP Tool Error: {error_content}"}

            # Success - Process content
            if mcp_response.content:
                try:
                    # --- Assume the result dictionary is JSON encoded in the text of the first part ---
                    first_part = mcp_response.content[0]

                    # Check if it's a TextContent part
                    if hasattr(first_part, 'type') and first_part.type == 'text' and hasattr(first_part, 'text'):
                        result_text = first_part.text
                        logger.debug(f"MCP Client Logic: Attempting to parse JSON from TextContent: {result_text}")
                        result_dict = json.loads(result_text) # <-- Parse JSON string

                        if isinstance(result_dict, dict):
                            # Now check the parsed dictionary
                            if "error" in result_dict:
                                logger.warning(f"MCP Client Logic: MCP tool logic reported error: {result_dict['error']}")
                            else:
                                logger.info("MCP Client Logic: MCP tool call successful.")
                            return result_dict # Return the parsed dictionary
                        else:
                            logger.error(f"MCP Client Logic: Parsed content is not a dictionary: {result_dict}")
                            return {"error": "MCP tool returned invalid JSON structure."}
                    # Optional: Add checks here if the tool might return DataPart directly
                    # elif isinstance(first_part, DataPart) and isinstance(first_part.data, dict):
                    #     logger.debug(f"MCP Client Logic: Extracted result dict directly from DataPart.")
                    #     result_dict = first_part.data
                    #     # ... check for error or price key ...
                    #     return result_dict
                    else:
                        # Handle cases where content[0] is not TextContent or expected type
                        logger.error(f"MCP Client Logic: Unexpected content part type in response: {type(first_part)}")
                        return {"error": f"Unexpected content type received from MCP tool: {type(first_part).__name__}"}

                except json.JSONDecodeError as json_e:
                     logger.error(f"MCP Client Logic: Failed to parse JSON from tool result text: '{result_text}'. Error: {json_e}")
                     return {"error": "MCP tool returned non-JSON text."}
                except (IndexError, AttributeError, TypeError) as e:
                    logger.error(f"MCP Client Logic: Error accessing expected content structure: {e}", exc_info=True)
                    return {"error": "Invalid content structure received from MCP tool."} # Your original error message fits here now
            else:
                # Handle case where successful response has no content
                logger.error("MCP Client Logic: Successful MCP response has no content.")
                return {"error": "Missing content in successful MCP tool response."}
        except asyncio.TimeoutError:
             logger.error(f"MCP Client Logic: Timeout occurred during MCP interaction for {symbol}.", exc_info=True)
             return {"error": f"Timeout communicating with MCP server for {symbol}."}
        except Exception as e:
            logger.error(f"MCP Client Logic: Unexpected error during MCP interaction for {symbol}: {e}", exc_info=True)
            return {"error": f"Failed to communicate with MCP server: {str(e)}"}

    logger.debug("MCP Client Logic: stdio_client context exited, subprocess terminated.")
    return {"error": "MCP interaction context finished unexpectedly."}

# --- Main Execution Block ---
# (Keep the __main__ block as is, it's only for running this file standalone, which we don't do)
# if __name__ == "__main__":
#     # Example of running this logic standalone (for testing mcp_client_logic itself)
#     async def test_mcp_call():
#         mcp_server_path = os.getenv("STOCK_MCP_SERVER_PATH") # Needs env var set
#         if not mcp_server_path:
#             print("Error: STOCK_MCP_SERVER_PATH env var not set for testing.")
#             return
#         result = await call_mcp_get_stock_price("AAPL", mcp_server_path)
#         print("Standalone Test Result:", result)
#     asyncio.run(test_mcp_call())