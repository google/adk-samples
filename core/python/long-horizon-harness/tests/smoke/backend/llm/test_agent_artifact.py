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

"""LLM-driven: agent routes save/load asks through the ``artifact`` tool.

Targets the same regression that ``test_artifact_save_load.py`` locks in
at the tool level — the just-fixed ``Blob.display_name`` bug — but
catches a different failure mode: the agent forgetting to call
``artifact`` at all, or routing through ``write_file`` instead.
"""

from __future__ import annotations

import pytest

from tests.smoke.backend.llm.conftest import function_call_names

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_agent_invokes_artifact_save_for_save_request(
    llm_runner, active_env, unique_name
) -> None:
    fname = f"{unique_name}.txt"
    abs_path = active_env.working_dir / fname
    await active_env.write_file(abs_path, b"hello smoke")

    events = await llm_runner.run_once(
        f"Use the `artifact` tool with action=save and path={fname} "
        "to save that workspace file as a downloadable artifact."
    )

    names = function_call_names(events)
    assert "artifact" in names, (
        f"expected agent to call `artifact`; got tool calls: {names}"
    )
