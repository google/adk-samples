"""Local helper tools for the Crypto Payroll Agent.

These pure-Python helpers extend the Spraay batch tools with conveniences
the model will reach for naturally:

* `lookup_token_info` — resolve a token symbol (e.g. "USDC") to its
  canonical address and decimals on Base, so the user never has to look
  them up.

* `split_pool_proportionally` — divide a total token amount across N
  weights, rounding-safe so the per-recipient amounts sum exactly to the
  pool total. Pairs naturally with `spraay_batch_token_variable` and
  `spraay_batch_eth_variable`.

Both functions are plain callables; ADK adapts them to FunctionTools when
they are listed in `Agent(tools=[...])`.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal, getcontext
from typing import Any

from ..config import BASE_TOKEN_REGISTRY

# Decimal precision wide enough to handle 18-decimal token math.
getcontext().prec = 50


def lookup_token_info(symbol_or_address: str) -> dict[str, Any]:
    """Return canonical metadata for a Base ERC-20 token.

    Accepts a token symbol (case-insensitive, e.g. "usdc", "USDC", "WETH")
    or a raw 0x address. When an address is supplied, the registry is
    searched for a match; if not found, the address is echoed back with
    `decimals` unknown — the caller must then ask the user for decimals
    explicitly.

    Args:
        symbol_or_address: A token symbol or a 0x... address.

    Returns:
        A dict with shape:
            {
                "found": bool,
                "symbol": str,        # canonical display symbol
                "address": str,       # 0x address on Base
                "decimals": int | None,
                "source": "registry" | "unknown",
                "note": str,          # human-readable note
            }
    """
    query = (symbol_or_address or "").strip()

    if query.lower().startswith("0x") and len(query) == 42:
        for token in BASE_TOKEN_REGISTRY.values():
            if token["address"].lower() == query.lower():
                return {
                    "found": True,
                    "symbol": token["symbol"],
                    "address": token["address"],
                    "decimals": token["decimals"],
                    "source": "registry",
                    "note": (
                        f"Matched address to {token['symbol']} on Base."
                    ),
                }
        return {
            "found": False,
            "symbol": "",
            "address": query,
            "decimals": None,
            "source": "unknown",
            "note": (
                "Address not in the bundled registry. Ask the user for "
                "the token's decimals before constructing a transaction."
            ),
        }

    key = query.upper()
    token = BASE_TOKEN_REGISTRY.get(key)
    if token is not None:
        return {
            "found": True,
            "symbol": token["symbol"],
            "address": token["address"],
            "decimals": token["decimals"],
            "source": "registry",
            "note": f"Resolved {token['symbol']} on Base.",
        }

    return {
        "found": False,
        "symbol": symbol_or_address,
        "address": "",
        "decimals": None,
        "source": "unknown",
        "note": (
            f"'{symbol_or_address}' is not a recognized symbol in the "
            "bundled Base token registry. Ask the user for the canonical "
            "contract address and decimals before proceeding."
        ),
    }


def split_pool_proportionally(
    total_amount: str,
    weights: list[float],
    decimals: int = 6,
) -> dict[str, Any]:
    """Divide a token pool across recipients by integer weights.

    Distribution is rounding-safe: the returned per-recipient amounts sum
    *exactly* to `total_amount` (any rounding dust is added to the largest
    share). Amounts are returned as decimal strings suitable to pass to
    `spraay_batch_token_variable` (which expects human-unit strings).

    Args:
        total_amount: Pool total in human units (e.g. "10000" for 10,000 USDC).
        weights: List of non-negative weights, one per recipient.
        decimals: Token decimals used to clamp precision (default 6 for USDC).

    Returns:
        A dict:
            {
                "ok": bool,
                "amounts": list[str],       # ordered to match `weights`
                "total_distributed": str,   # sums to `total_amount`
                "error": str | None,
            }
    """
    if not weights:
        return {
            "ok": False,
            "amounts": [],
            "total_distributed": "0",
            "error": "weights must be a non-empty list.",
        }

    if any(w < 0 for w in weights):
        return {
            "ok": False,
            "amounts": [],
            "total_distributed": "0",
            "error": "weights must be non-negative.",
        }

    try:
        total = Decimal(total_amount)
    except Exception:  # noqa: BLE001
        return {
            "ok": False,
            "amounts": [],
            "total_distributed": "0",
            "error": f"total_amount '{total_amount}' is not a valid number.",
        }

    if total <= 0:
        return {
            "ok": False,
            "amounts": [],
            "total_distributed": "0",
            "error": "total_amount must be positive.",
        }

    weight_sum = Decimal(sum(weights))
    if weight_sum == 0:
        return {
            "ok": False,
            "amounts": [],
            "total_distributed": "0",
            "error": "weights sum to zero.",
        }

    # Precision used for rounding per-recipient amounts.
    quant = Decimal(10) ** -decimals
    one_unit = quant  # smallest distributable amount at this precision

    # First pass: floor each share to `decimals` precision. Record the
    # fractional remainder so we can apportion the dust fairly using the
    # largest-remainder method.
    raw_shares = [Decimal(str(w)) / weight_sum * total for w in weights]
    floored = [s.quantize(quant, rounding=ROUND_DOWN) for s in raw_shares]
    remainders = [
        (i, raw_shares[i] - floored[i]) for i in range(len(weights))
    ]

    shares = list(floored)
    dust = total - sum(shares)

    if dust > 0:
        # How many quanta of dust to distribute?
        units_to_distribute = int((dust / one_unit).quantize(Decimal("1")))

        # Sort recipients by remainder (desc), break ties by weight (desc).
        remainders.sort(
            key=lambda x: (x[1], Decimal(str(weights[x[0]]))),
            reverse=True,
        )

        for k in range(units_to_distribute):
            target_idx = remainders[k % len(remainders)][0]
            shares[target_idx] += one_unit

    return {
        "ok": True,
        "amounts": [str(s) for s in shares],
        "total_distributed": str(sum(shares)),
        "error": None,
    }
