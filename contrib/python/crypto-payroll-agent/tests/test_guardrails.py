"""Tests for the before_tool_callback spend ceilings and batch limits.

All offline: the callback is a pure function of the tool name and the
argument dict, so nothing here touches a chain, an RPC endpoint, or a
signing key.
"""

from __future__ import annotations

from types import SimpleNamespace

# Ceilings come from .env.example via conftest: 10000 USD, 5 ETH.
USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
WETH = "0x4200000000000000000000000000000000000006"

ADDR_A = "0x9f2c000000000000000000000000000000000000"
ADDR_B = "0x4ab7000000000000000000000000000000000000"


def _tool(name: str) -> SimpleNamespace:
    """Stand in for the ADK BaseTool the callback receives."""
    return SimpleNamespace(name=name)


def _call(tool_name: str, **args):
    from crypto_payroll_agent.guardrails import enforce_batch_limits

    return enforce_batch_limits(
        tool=_tool(tool_name), args=args, tool_context=None
    )


def test_allows_token_batch_under_usd_ceiling():
    # 2 x 250 USDC = 500, well under the 10000 USD ceiling.
    assert (
        _call(
            "spraay_batch_token",
            token_address=USDC,
            recipients=[ADDR_A, ADDR_B],
            amount_per_recipient="250",
            token_decimals=6,
        )
        is None
    )


def test_allows_eth_batch_under_eth_ceiling():
    # 2 x 0.01 ETH = 0.02, well under the 5 ETH ceiling.
    assert (
        _call(
            "spraay_batch_eth",
            recipients=[ADDR_A, ADDR_B],
            amount_per_recipient_eth="0.01",
        )
        is None
    )


def test_blocks_token_batch_over_usd_ceiling():
    # 2 x 8000 USDC = 16000, over the 10000 USD ceiling.
    result = _call(
        "spraay_batch_token",
        token_address=USDC,
        recipients=[ADDR_A, ADDR_B],
        amount_per_recipient="8000",
        token_decimals=6,
    )
    assert result is not None
    assert result["status"] == "error"
    assert "PAYROLL_MAX_BATCH_USD" in result["error"]


def test_blocks_eth_batch_over_eth_ceiling():
    # 4 + 3 = 7 ETH, over the 5 ETH ceiling.
    result = _call(
        "spraay_batch_eth_variable",
        recipients=[ADDR_A, ADDR_B],
        amounts_eth=["4", "3"],
    )
    assert result is not None
    assert result["status"] == "error"
    assert "PAYROLL_MAX_BATCH_ETH" in result["error"]


def test_blocks_non_pegged_token_even_under_ceiling():
    # 2 WETH is far under 10000, but WETH has no offline USD value.
    result = _call(
        "spraay_batch_token",
        token_address=WETH,
        recipients=[ADDR_A, ADDR_B],
        amount_per_recipient="1",
        token_decimals=18,
    )
    assert result is not None
    assert result["status"] == "error"
    assert "not a $1-pegged token" in result["error"]
    # The message must name the customization point.
    assert "enforce_batch_limits" in result["error"]


def test_blocks_more_than_200_recipients():
    recipients = [f"0x{i:040x}" for i in range(201)]
    result = _call(
        "spraay_batch_eth",
        recipients=recipients,
        amount_per_recipient_eth="0.0001",
    )
    assert result is not None
    assert result["status"] == "error"
    assert "201 recipients" in result["error"]


def test_blocks_malformed_recipient_address():
    result = _call(
        "spraay_batch_token",
        token_address=USDC,
        recipients=[ADDR_A, "alice.eth"],
        amount_per_recipient="10",
        token_decimals=6,
    )
    assert result is not None
    assert result["status"] == "error"
    assert "alice.eth" in result["error"]


def test_ignores_tools_that_are_not_spraay_batches():
    assert _call("lookup_token_info", symbol_or_address="USDC") is None


def test_agent_wires_the_callback():
    from crypto_payroll_agent import root_agent
    from crypto_payroll_agent.guardrails import enforce_batch_limits

    assert root_agent.before_tool_callback is enforce_batch_limits
