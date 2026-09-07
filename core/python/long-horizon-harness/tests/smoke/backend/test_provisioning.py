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

"""``shared_environment`` is reachable and round-trips basic IO.

If this test fails, no other backend smoke test can pass — the harness
is broken. Kept minimal so a failure points at provisioning, not at the
artifact or tool layers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk.environment import BaseEnvironment

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_environment_initializes_and_reads_back_a_write(
    shared_environment: BaseEnvironment, unique_name: str
) -> None:
    payload = b"smoke-provisioning-payload"
    target = shared_environment.working_dir / f"{unique_name}.txt"

    await shared_environment.write_file(target, payload)
    read_back = await shared_environment.read_file(target)

    assert read_back == payload


async def test_environment_working_dir_is_a_concrete_path(
    shared_environment: BaseEnvironment,
) -> None:
    assert isinstance(shared_environment.working_dir, Path)
    assert str(shared_environment.working_dir)
