# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test-wide setup. Runs before pytest collects any test module so mocks
are in place before app.* is imported (some app modules do real GCP work
at import time).
"""

import sys
from pathlib import Path

# Make sibling `_env_stubs.py` importable without needing tests/ to be a
# proper package (which would collide with pytest's importlib mode).
sys.path.insert(0, str(Path(__file__).parent))

from _env_stubs import install as _install_test_stubs

_install_test_stubs()
