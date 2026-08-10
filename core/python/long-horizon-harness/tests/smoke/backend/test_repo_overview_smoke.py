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

"""``repo_overview()`` generated-script walker works against the env.

Direct tool-call rather than LLM-driven — same pattern as
``test_terminal_tool.py``. The smoke contract is "the embedded Python
walker runs through ``env.execute`` and reads the real (sandbox)
filesystem": we seed a tiny project under the env root via
``env.write_file``, then assert ecosystem detection, entrypoint parsing,
and a depth-limited tree come back. LLM behavior is covered by
``tests/eval``.
"""

from __future__ import annotations

import pytest
from google.adk.environment import BaseEnvironment

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_walker_orients_on_seeded_project(
    active_env: BaseEnvironment,
) -> None:
    from horizon.tools.repo_overview import repo_overview

    root = active_env.working_dir.resolve()
    await active_env.write_file(
        root / "pyproject.toml",
        "[project]\nname = 'smoke-proj'\n\n"
        "[project.scripts]\nsmoke-cli = 'smoke_proj.cli:main'\n",
    )
    await active_env.write_file(
        root / "src" / "smoke_proj" / "cli.py", "def main():\n    pass\n"
    )
    await active_env.write_file(
        root / "src" / "smoke_proj" / "core.py", "VALUE = 1\n"
    )

    result = await repo_overview(path=str(root), depth=2)

    assert result["success"] is True
    assert "Python" in result["ecosystems"]
    assert "pyproject.toml" in result["dependency_files"]
    assert any("smoke-cli" in e for e in result["entrypoints"])
    assert any("smoke_proj.cli:main" in e for e in result["entrypoints"])
    assert "src/" in result["overview"]
    assert "smoke_proj/" in result["overview"]
    assert result["depth"] == 2


async def test_depth_limit_is_enforced_in_env(
    active_env: BaseEnvironment,
) -> None:
    from horizon.tools.repo_overview import repo_overview

    root = active_env.working_dir.resolve()
    await active_env.write_file(
        root / "deeptree" / "lvl1" / "lvl2" / "buried.py", "x = 1\n"
    )

    result = await repo_overview(path=str(root), depth=1)

    assert result["success"] is True
    assert "deeptree/" in result["overview"]
    assert "buried.py" not in result["overview"]
