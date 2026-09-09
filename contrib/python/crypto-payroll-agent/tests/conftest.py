"""Provide required environment variables for offline test runs.

The values are not duplicated here: every variable declared in the
recipe's `.env.example` is loaded from it, so that file stays the single
source of truth. Some of its values are `<TODO: update-this-value>`
placeholders — that is fine, and deliberate: the offline suite never
reaches a real credential, so a placeholder proves the tests do not
quietly depend on one. `setdefault` means a real `.env` or an exported
variable always wins.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

RECIPE_ROOT = Path(__file__).resolve().parent.parent

for key, value in dotenv_values(RECIPE_ROOT / ".env.example").items():
    if value:
        os.environ.setdefault(key, value)
