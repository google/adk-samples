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

"""before_model_callback rewrites delegate/agent tool descriptions live."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

pytestmark = pytest.mark.asyncio


def _write_skill(root: Path, name: str, description: str) -> None:
    from horizon.tools.skill_reload import host_mirror_dir

    frontmatter = yaml.safe_dump(
        {"name": name, "description": description}, sort_keys=True
    )
    content = f"---\n{frontmatter}---\n\nBody for {name}.\n"
    mirror_dir = host_mirror_dir(root) / name
    mirror_dir.mkdir(parents=True, exist_ok=True)
    (mirror_dir / "SKILL.md").write_text(content, encoding="utf-8")


@pytest.fixture()
def skills_home(tmp_path: Path):
    from horizon.environment import LocalEnvironment
    from horizon.environment_context import (
        clear_active_environment,
        set_active_environment,
    )
    from horizon.tools.skill_reload import host_mirror_dir

    set_active_environment(LocalEnvironment(working_dir=tmp_path))
    try:
        yield tmp_path
    finally:
        clear_active_environment()
        shutil.rmtree(host_mirror_dir(tmp_path), ignore_errors=True)


def _make_request(*tool_names: str):
    from google.genai import types

    fds = [
        types.FunctionDeclaration(name=n, description=f"BASE {n} description.")
        for n in tool_names
    ]
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=fds)]
    )

    class _Req:
        pass

    req = _Req()
    req.config = config
    return req


class _Cb:
    state: ClassVar[dict[str, Any]] = {}


def _desc(req: Any, name: str) -> str:
    for tool in req.config.tools or []:
        for fd in tool.function_declarations or []:
            if fd.name == name:
                return fd.description
    raise AssertionError(name)


async def test_rewrites_delegate_and_agent_with_skills(
    skills_home: Path,
) -> None:
    _write_skill(skills_home, "gws-drive", "Drive enumeration helper.")
    from horizon.subagents.descriptions import (
        make_subagent_description_callback,
    )

    cb = make_subagent_description_callback()
    req = _make_request("delegate", "agent", "read_file")
    await cb(_Cb(), req)

    delegate_desc = _desc(req, "delegate")
    assert delegate_desc.startswith("BASE delegate description.")
    assert "## Skills you can pass to a child" in delegate_desc
    assert "- gws-drive: Drive enumeration helper." in delegate_desc
    assert "## Child profiles" in delegate_desc
    # agent gets the same treatment.
    assert "## Child profiles" in _desc(req, "agent")
    # other tools are untouched.
    assert _desc(req, "read_file") == "BASE read_file description."


async def test_empty_catalog_shows_placeholder(monkeypatch) -> None:
    """When the catalog is genuinely empty, a placeholder line is shown
    instead of an empty block."""
    from horizon.subagents import descriptions as descriptions_mod

    monkeypatch.setattr(
        descriptions_mod, "load_session_skill_catalog", lambda: {}
    )
    block = descriptions_mod.render_skills_block()
    assert "_No skills are currently loaded._" in block


async def test_block_lists_loaded_skills(skills_home: Path) -> None:
    """The block lists the skills actually resolvable this session
    (built-ins are always loaded; user-mirror skills add to them)."""
    _write_skill(skills_home, "gws-drive", "Drive enumeration helper.")
    from horizon.subagents.descriptions import (
        make_subagent_description_callback,
    )

    cb = make_subagent_description_callback()
    req = _make_request("delegate")
    await cb(_Cb(), req)
    desc = _desc(req, "delegate")
    assert "- gws-drive: Drive enumeration helper." in desc
    assert "_No skills are currently loaded._" not in desc


async def test_idempotent_across_turns(skills_home: Path) -> None:
    """Calling the callback twice must not double-append the dynamic block."""
    _write_skill(skills_home, "s1", "First skill.")
    from horizon.subagents.descriptions import (
        make_subagent_description_callback,
    )

    cb = make_subagent_description_callback()
    req = _make_request("delegate")
    await cb(_Cb(), req)
    first = _desc(req, "delegate")
    await cb(_Cb(), req)
    second = _desc(req, "delegate")
    assert first == second
    assert second.count("## Skills you can pass to a child") == 1


async def test_handles_request_with_no_tools() -> None:
    from horizon.subagents.descriptions import (
        make_subagent_description_callback,
    )

    cb = make_subagent_description_callback()

    class _Req:
        class config:
            tools = None

    # Must not raise.
    await cb(_Cb(), _Req())
