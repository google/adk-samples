# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0
"""Make this skill's scripts/ directory importable from its tests.

Keeping the path shim inside the skill (rather than in the repo-root pytest
config) preserves the skill as a self-contained, portable bundle.
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
