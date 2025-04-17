from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams

mcp_exit_stack = None

# Asynchronous function to get tools from a remote MCP Agent server
async def get_mcp_tools():
    global mcp_exit_stack
    tools, mcp_exit_stack = await MCPToolset.from_server(
        connection_params=SseServerParams(url="")  # Fill in the mcp URL here
    )
    return tools

# Asynchronous function to gracefully close the MCP connection
async def close_mcp_connection():
    if mcp_exit_stack:
        await mcp_exit_stack.aclose()