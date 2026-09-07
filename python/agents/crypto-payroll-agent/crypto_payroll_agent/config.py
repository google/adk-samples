"""Configuration for the Crypto Payroll Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TypedDict


class TokenInfo(TypedDict):
    """Metadata for an ERC-20 token on Base."""

    symbol: str
    address: str
    decimals: int


# Base mainnet token registry.
# Addresses verified against https://basescan.org as of 2025.
# Verify each entry before relying on it in production.
BASE_TOKEN_REGISTRY: dict[str, TokenInfo] = {
    "USDC": {
        "symbol": "USDC",
        "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "decimals": 6,
    },
    "USDBC": {
        "symbol": "USDbC",
        "address": "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA",
        "decimals": 6,
    },
    "WETH": {
        "symbol": "WETH",
        "address": "0x4200000000000000000000000000000000000006",
        "decimals": 18,
    },
    "CBETH": {
        "symbol": "cbETH",
        "address": "0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22",
        "decimals": 18,
    },
    "CBBTC": {
        "symbol": "cbBTC",
        "address": "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",
        "decimals": 8,
    },
    "DAI": {
        "symbol": "DAI",
        "address": "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",
        "decimals": 18,
    },
    "AERO": {
        "symbol": "AERO",
        "address": "0x940181a94A35A4569E4529A3CDfB74e38FD98631",
        "decimals": 18,
    },
}


@dataclass(frozen=True)
class Config:
    """Runtime config sourced from environment variables."""

    # Model
    model: str = os.getenv("PAYROLL_AGENT_MODEL", "gemini-2.5-flash")

    # Safety ceiling for a single batch run (in USD)
    max_batch_usd: Decimal = Decimal(
        os.getenv("PAYROLL_MAX_BATCH_USD", "10000")
    )

    # Agent metadata
    agent_name: str = "crypto_payroll_agent"
    app_name: str = "Crypto Payroll Agent"

    # Token registry (immutable copy)
    token_registry: dict[str, TokenInfo] = field(
        default_factory=lambda: dict(BASE_TOKEN_REGISTRY)
    )


CONFIG = Config()
