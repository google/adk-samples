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

"""Suite-wide gates and per-test overrides for the smoke layer.

Smoke tests provision real resources (a sandbox container, an HTTP
server). They are opt-in via ``RUN_SMOKE=1`` so the default ``uv run
pytest`` stays hermetic and free.

The repo-level ``tests/conftest.py`` autouses ``_scoped_environment`` to
bind a fresh ``LocalEnvironment`` per test. Smoke tests need the
opposite: a single shared environment that survives across the whole
session. Overridden to a no-op here so each smoke test can install its
own environment without ``_scoped_environment``'s teardown clobbering it
on the next test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-mark every test under ``tests/smoke`` with the ``RUN_SMOKE``
    skip gate so individual smoke tests don't have to remember it."""
    if os.environ.get("RUN_SMOKE"):
        return
    skip = pytest.mark.skip(reason="set RUN_SMOKE=1 to enable smoke tests")
    for item in items:
        if "tests/smoke" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _scoped_environment() -> Iterator[None]:
    """Override the repo-level autouse so smoke tests own env binding."""
    yield
