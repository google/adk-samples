import os

from google.adk.agents import Agent
from google.adk.tools import MCPToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters

root_agent = Agent(
    name="antom_payment_agent",
    model="gemini-2.5-flash",
    description=(
        "Agent creates payment links for merchants, queries payment result details。"
    ),
    instruction=(
        "You are an Antom payment agent who can help users create payment links and query payment result details."
        "Regarding RequestId, you generate it randomly"
        "And you can describe the description of the user creating the order in one sentence"
        "when refund get the order details and paymentId based on the paymentRequest ID used when creating "
        "the payment method by the payment agent. "
        "If the merchant specifies a refund amount, a full refund will be made in the order currency by default."
    ),
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="uvx",
                    args=["ant-intl-antom-mcp"],
                    # Pass the API key as an environment variable to the npx process
                    # This is how the MCP server for Antom payment expects the key.
                    env={
                        "GATEWAY_URL": os.getenv("GATEWAY_URL"),
                        "CLIENT_ID": os.getenv("CLIENT_ID"),
                        "MERCHANT_PRIVATE_KEY": os.getenv(
                            "MERCHANT_PRIVATE_KEY"
                        ),
                        "ALIPAY_PUBLIC_KEY": os.getenv("ALIPAY_PUBLIC_KEY"),
                        "PAYMENT_REDIRECT_URL": os.getenv(
                            "PAYMENT_REDIRECT_URL"
                        ),
                        "PAYMENT_NOTIFY_URL": os.getenv("PAYMENT_NOTIFY_URL"),
                    },
                ),
            ),
        )
    ],
)
