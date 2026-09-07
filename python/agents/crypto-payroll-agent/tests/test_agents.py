"""Unit tests for the Crypto Payroll Agent.

The local helpers are tested directly. The agent assembly smoke test
imports the agent and verifies its tool roster. The Spraay batch tools
themselves are integration-tested against `google-adk-community` upstream
and are not re-tested here.
"""

from __future__ import annotations

from decimal import Decimal

import pytest


# --------------------------------------------------------------------------
# lookup_token_info
# --------------------------------------------------------------------------
def test_lookup_token_info_resolves_known_symbol():
    from crypto_payroll_agent.tools.helpers import lookup_token_info

    result = lookup_token_info("USDC")
    assert result["found"] is True
    assert result["symbol"] == "USDC"
    assert result["decimals"] == 6
    assert result["address"].startswith("0x833589")


def test_lookup_token_info_is_case_insensitive():
    from crypto_payroll_agent.tools.helpers import lookup_token_info

    for query in ("usdc", "Usdc", "USDC"):
        assert lookup_token_info(query)["decimals"] == 6


def test_lookup_token_info_resolves_known_address():
    from crypto_payroll_agent.tools.helpers import lookup_token_info

    weth = "0x4200000000000000000000000000000000000006"
    result = lookup_token_info(weth)
    assert result["found"] is True
    assert result["symbol"] == "WETH"
    assert result["decimals"] == 18


def test_lookup_token_info_returns_not_found_for_unknown_symbol():
    from crypto_payroll_agent.tools.helpers import lookup_token_info

    result = lookup_token_info("MYSTERY")
    assert result["found"] is False
    assert result["decimals"] is None
    assert "not a recognized symbol" in result["note"]


def test_lookup_token_info_returns_not_found_for_unknown_address():
    from crypto_payroll_agent.tools.helpers import lookup_token_info

    unknown = "0x1111111111111111111111111111111111111111"
    result = lookup_token_info(unknown)
    assert result["found"] is False
    assert result["address"] == unknown
    assert "decimals" in result["note"]


# --------------------------------------------------------------------------
# split_pool_proportionally
# --------------------------------------------------------------------------
def test_split_pool_simple_equal_weights():
    from crypto_payroll_agent.tools.helpers import split_pool_proportionally

    result = split_pool_proportionally("300", [1, 1, 1], decimals=6)
    assert result["ok"] is True
    # All three shares are equal and reconcile exactly to the total.
    assert len({Decimal(a) for a in result["amounts"]}) == 1
    assert Decimal(result["amounts"][0]) == Decimal("100")
    assert Decimal(result["total_distributed"]) == Decimal("300")


def test_split_pool_proportional_weights():
    from crypto_payroll_agent.tools.helpers import split_pool_proportionally

    result = split_pool_proportionally("10000", [120, 80, 50], decimals=6)
    assert result["ok"] is True
    # Reconciles exactly to the pool total.
    assert Decimal(result["total_distributed"]) == Decimal("10000")
    # Largest-weight recipient gets the largest share.
    amounts = [Decimal(a) for a in result["amounts"]]
    assert amounts[0] > amounts[1] > amounts[2]


def test_split_pool_rounding_dust_goes_to_largest_weight():
    from crypto_payroll_agent.tools.helpers import split_pool_proportionally

    # 100 / 3 with 6 decimals leaves rounding dust of 0.000001 USDC.
    result = split_pool_proportionally("100", [1, 1, 1], decimals=6)
    assert result["ok"] is True
    assert Decimal(result["total_distributed"]) == Decimal("100")


def test_split_pool_rejects_empty_weights():
    from crypto_payroll_agent.tools.helpers import split_pool_proportionally

    result = split_pool_proportionally("100", [], decimals=6)
    assert result["ok"] is False
    assert "non-empty" in result["error"]


def test_split_pool_rejects_negative_weights():
    from crypto_payroll_agent.tools.helpers import split_pool_proportionally

    result = split_pool_proportionally("100", [1, -1, 2], decimals=6)
    assert result["ok"] is False
    assert "non-negative" in result["error"]


def test_split_pool_rejects_zero_total():
    from crypto_payroll_agent.tools.helpers import split_pool_proportionally

    result = split_pool_proportionally("0", [1, 1], decimals=6)
    assert result["ok"] is False
    assert "positive" in result["error"]


def test_split_pool_rejects_zero_weight_sum():
    from crypto_payroll_agent.tools.helpers import split_pool_proportionally

    result = split_pool_proportionally("100", [0, 0, 0], decimals=6)
    assert result["ok"] is False
    assert "zero" in result["error"]


def test_split_pool_18_decimals_for_eth_amounts():
    from crypto_payroll_agent.tools.helpers import split_pool_proportionally

    # Splitting 1 ETH among 3 with 18 decimals.
    result = split_pool_proportionally("1", [1, 1, 1], decimals=18)
    assert result["ok"] is True
    assert Decimal(result["total_distributed"]) == Decimal("1")


# --------------------------------------------------------------------------
# Agent assembly smoke test
# --------------------------------------------------------------------------
def test_root_agent_has_expected_tools():
    """Agent imports cleanly and exposes all six tools."""
    from crypto_payroll_agent import root_agent

    tool_names = {
        getattr(t, "__name__", None) or getattr(t, "name", "")
        for t in root_agent.tools
    }

    # Local helpers
    assert "lookup_token_info" in tool_names
    assert "split_pool_proportionally" in tool_names

    # Spraay batch tools
    assert "spraay_batch_eth" in tool_names
    assert "spraay_batch_token" in tool_names
    assert "spraay_batch_eth_variable" in tool_names
    assert "spraay_batch_token_variable" in tool_names

    assert root_agent.name == "crypto_payroll_agent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
