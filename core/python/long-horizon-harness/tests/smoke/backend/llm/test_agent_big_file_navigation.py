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

"""LLM-driven: agent navigates a large file past the first read window.

We seed a file longer than ``read_file``'s default 500-line window, planting a
unique answer token deep in a late section. The agent can only answer by
continuing past the first window (``read_file`` offset/limit, the continuation
trailer, or grep). We assert the planted token appears in the final answer —
a deterministic content check, not a free-text assertion.
"""

from __future__ import annotations

import pytest
from google.adk.environment import BaseEnvironment

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _final_text(events: list) -> str:
    chunks: list[str] = []
    for event in events:
        content = getattr(event, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


async def test_agent_reads_late_section_of_large_file(
    llm_runner, active_env: BaseEnvironment, unique_name: str
) -> None:
    root = active_env.working_dir.resolve()
    target = f"big_{unique_name}.txt"
    answer = f"SECRET_{unique_name}_ZZ"

    # 1200 lines (> the 500-line default window). The answer is on line ~1100,
    # well past the first read window.
    lines = [f"line-{i:05d}: filler" for i in range(1200)]
    lines[1100] = f"line-01100: the magic password is {answer}"
    await active_env.write_file(root / target, "\n".join(lines) + "\n")

    prompt = (
        f"The file '{target}' is long. Somewhere in it there is a line that "
        "states 'the magic password is ...'. Read through the file as needed "
        "and tell me the magic password."
    )

    events = await llm_runner.run_once(prompt)

    assert answer in _final_text(events), (
        "agent did not retrieve the answer from the late section — likely "
        "only read the first window"
    )
