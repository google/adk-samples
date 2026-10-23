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

"""LLM-driven: agent edits a file via the fuzzy ``patch`` path.

We seed a file through the env interface, then ask the agent to change a function,
handing it a snippet whose whitespace drifts from the file's exact text. The
fuzzy replacer ladder in ``patch`` must still apply the edit. We assert the
durable side effect — the file content changed through the env write path —
not the agent's wording. The replacement marker is a deterministic token.
"""

from __future__ import annotations

import pytest
from google.adk.environment import BaseEnvironment

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_agent_applies_fuzzy_edit_to_seeded_file(
    llm_runner, active_env: BaseEnvironment, unique_name: str
) -> None:
    root = active_env.working_dir.resolve()
    target = f"fuzzy_{unique_name}.py"
    original = "def total(items):\n    return sum(items)\n"
    await active_env.write_file(root / target, original)

    marker = f"DOUBLED_{unique_name}"
    # The snippet handed to the agent drifts in spacing from the real line,
    # so a non-fuzzy exact match would miss.
    prompt = (
        f"In the file '{target}', change the body of the `total` function. "
        "It currently has roughly this line:\n"
        "    return  sum( items )\n"
        "Change it so the function returns 2 * sum(items) and add a comment "
        f"line '# {marker}' above the return. Use the patch tool to make the "
        "edit in place."
    )

    await llm_runner.run_once(prompt)

    raw = await active_env.read_file(root / target)
    updated = raw.decode("utf-8")
    assert updated != original, "file was not modified"
    assert marker in updated, (
        "expected the agent's edit to land in the file via the fuzzy patch path"
    )
