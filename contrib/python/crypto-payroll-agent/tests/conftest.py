"""Provide required environment variables for offline test runs.

The values are not duplicated here: they are read from the recipe's
`.env.example`, which is the single source of truth for every variable the
recipe declares. `setdefault` means a real `.env` or an exported variable
always wins.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

RECIPE_ROOT = Path(__file__).resolve().parent.parent

for key, value in dotenv_values(RECIPE_ROOT / ".env.example").items():
    if key.startswith("PAYROLL_") and value:
        os.environ.setdefault(key, value)
