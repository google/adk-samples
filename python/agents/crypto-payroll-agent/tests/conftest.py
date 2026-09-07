"""Provide required environment variables for offline test runs."""

import os

os.environ.setdefault("PAYROLL_AGENT_MODEL", "gemini-3.5-flash")
os.environ.setdefault("PAYROLL_MAX_BATCH_USD", "10000")