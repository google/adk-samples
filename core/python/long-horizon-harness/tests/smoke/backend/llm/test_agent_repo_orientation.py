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

"""LLM-driven: agent orients on an unfamiliar workspace and names the entrypoint.

We seed a tiny project (``pyproject.toml`` with a unique ``[project.scripts]``
entry + a source tree) through the env interface, then ask the agent what the
project is and where its entrypoint lives. The script NAME is a deterministic
token we planted, so asserting the agent surfaces it is a contract check on
``repo_overview`` orientation, not an assertion on free-text phrasing.
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


async def test_agent_names_seeded_entrypoint(
    llm_runner, active_env: BaseEnvironment, unique_name: str
) -> None:
    root = active_env.working_dir.resolve()
    proj_dir = f"orient_{unique_name}"
    script_name = f"orient-cli-{unique_name}"
    pkg = "orient_pkg"

    await active_env.write_file(
        root / proj_dir / "pyproject.toml",
        "[project]\n"
        f"name = '{proj_dir}'\n\n"
        "[project.scripts]\n"
        f"{script_name} = '{pkg}.cli:main'\n",
    )
    await active_env.write_file(
        root / proj_dir / "src" / pkg / "cli.py",
        "def main():\n    print('hello')\n",
    )

    prompt = (
        f"There is a project under the '{proj_dir}/' directory in my "
        "workspace. Use your tools to orient yourself, then tell me what "
        "this project is and the name of its command-line entrypoint."
    )

    events = await llm_runner.run_once(prompt)

    assert script_name in _final_text(events), (
        f"agent did not name the seeded entrypoint {script_name!r}"
    )
