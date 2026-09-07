"""Root agent for the Crypto Payroll Agent sample."""

from google.adk.agents import LlmAgent

# Spraay batch tools merged in google/adk-python-community#95.
from google.adk_community.tools.spraay import (  # type: ignore[import]
    spraay_batch_eth,
    spraay_batch_eth_variable,
    spraay_batch_token,
    spraay_batch_token_variable,
)

from .config import CONFIG
from .prompt import ROOT_AGENT_INSTRUCTION
from .tools import lookup_token_info, split_pool_proportionally

root_agent = LlmAgent(
    name=CONFIG.agent_name,
    model=CONFIG.model,
    description=(
        "Batch-pays multiple recipients in ETH or ERC-20 tokens on Base "
        "via the Spraay protocol. Supports equal-amount payroll, variable-"
        "amount contractor pay, and proportional pool splits for airdrops "
        "and revenue sharing."
    ),
    instruction=ROOT_AGENT_INSTRUCTION.format(
        max_batch_usd=str(CONFIG.max_batch_usd),
    ),
    tools=[
        # Local helpers (must be called before the batch tools when needed).
        lookup_token_info,
        split_pool_proportionally,
        # Spraay batch tools (on-chain action).
        spraay_batch_eth,
        spraay_batch_token,
        spraay_batch_eth_variable,
        spraay_batch_token_variable,
    ],
)
