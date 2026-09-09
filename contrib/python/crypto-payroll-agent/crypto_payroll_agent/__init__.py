"""Crypto Payroll Agent.

An ADK sample agent demonstrating multi-recipient stablecoin payouts via the
Spraay community tools.
"""

from dotenv import load_dotenv

load_dotenv()

from .agent import root_agent  # noqa: E402

__all__ = ["root_agent"]
