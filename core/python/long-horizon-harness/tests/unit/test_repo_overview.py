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

"""Deterministic tests for the repo_overview tool.

Like test_file_ops.py::TestSearchFiles, the tool runs an embedded Python
script through the active env (LocalEnvironment installed by the autouse
_scoped_environment fixture). Tests build a temp workspace and assert on
the structured fields + stable text substrings — never on LLM output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.asyncio,
]


def _write(p: Path, text: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


class TestEcosystemDetection:
    async def test_detects_python_from_pyproject(self, tmp_path: Path) -> None:
        from horizon.tools.repo_overview import repo_overview

        _write(tmp_path / "pyproject.toml", "[project]\nname = 'x'\n")

        result = await repo_overview(path=str(tmp_path))

        assert result["success"] is True
        assert "Python" in result["ecosystems"]
        assert "pyproject.toml" in result["dependency_files"]

    async def test_detects_python_from_requirements(
        self, tmp_path: Path
    ) -> None:
        from horizon.tools.repo_overview import repo_overview

        _write(tmp_path / "requirements.txt", "requests\n")

        result = await repo_overview(path=str(tmp_path))

        assert result["success"] is True
        assert "Python" in result["ecosystems"]
        assert "requirements.txt" in result["dependency_files"]

    async def test_detects_node_and_package_manager(
        self, tmp_path: Path
    ) -> None:
        from horizon.tools.repo_overview import repo_overview

        _write(tmp_path / "package.json", '{"name":"x"}')
        _write(tmp_path / "pnpm-lock.yaml", "lockfileVersion: 6\n")

        result = await repo_overview(path=str(tmp_path))

        assert result["success"] is True
        assert "Node.js" in result["ecosystems"]
        assert "pnpm" in result["package_managers"]

    async def test_detects_go_and_rust(self, tmp_path: Path) -> None:
        from horizon.tools.repo_overview import repo_overview

        _write(tmp_path / "go.mod", "module x\n")
        _write(tmp_path / "Cargo.toml", "[package]\nname='x'\n")

        result = await repo_overview(path=str(tmp_path))

        assert result["success"] is True
        assert "Go" in result["ecosystems"]
        assert "Rust" in result["ecosystems"]

    async def test_python_listed_first_when_mixed(self, tmp_path: Path) -> None:
        """Python-biased: Python comes before Node when both present."""
        from horizon.tools.repo_overview import repo_overview

        _write(tmp_path / "package.json", '{"name":"x"}')
        _write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")

        result = await repo_overview(path=str(tmp_path))

        eco = result["ecosystems"]
        assert eco.index("Python") < eco.index("Node.js")


class TestEntrypoints:
    async def test_parses_pyproject_scripts(self, tmp_path: Path) -> None:
        from horizon.tools.repo_overview import repo_overview

        _write(
            tmp_path / "pyproject.toml",
            "[project]\nname='x'\n\n[project.scripts]\nmycli = 'x.cli:main'\n",
        )

        result = await repo_overview(path=str(tmp_path))

        assert any("mycli" in e for e in result["entrypoints"])
        assert any("x.cli:main" in e for e in result["entrypoints"])

    async def test_finds_main_dunder(self, tmp_path: Path) -> None:
        from horizon.tools.repo_overview import repo_overview

        _write(tmp_path / "__main__.py", "print('hi')\n")

        result = await repo_overview(path=str(tmp_path))

        assert any("__main__.py" in e for e in result["entrypoints"])

    async def test_finds_app_agent_convention(self, tmp_path: Path) -> None:
        from horizon.tools.repo_overview import repo_overview

        _write(tmp_path / "app" / "agent.py", "root_agent = None\n")

        result = await repo_overview(path=str(tmp_path))

        assert any("app/agent.py" in e for e in result["entrypoints"])

    async def test_parses_package_json_main_and_bin(
        self, tmp_path: Path
    ) -> None:
        from horizon.tools.repo_overview import repo_overview

        _write(
            tmp_path / "package.json",
            '{"name":"x","main":"index.js","bin":{"x":"cli.js"}}',
        )

        result = await repo_overview(path=str(tmp_path))

        joined = " ".join(result["entrypoints"])
        assert "index.js" in joined
        assert "cli.js" in joined or "x" in joined

    async def test_no_entrypoints_is_empty_list_not_error(
        self, tmp_path: Path
    ) -> None:
        from horizon.tools.repo_overview import repo_overview

        _write(tmp_path / "notes.txt", "hello\n")

        result = await repo_overview(path=str(tmp_path))

        assert result["success"] is True
        assert result["entrypoints"] == []


class TestStructureTree:
    async def test_tree_respects_default_depth_3(self, tmp_path: Path) -> None:
        from horizon.tools.repo_overview import repo_overview

        # depth 0:a/ 1:b/ 2:c/ 3:d/ — d is at level 3, must be excluded.
        _write(tmp_path / "a" / "b" / "c" / "d" / "deep.py", "x\n")

        result = await repo_overview(path=str(tmp_path))

        assert "a/" in result["overview"]
        assert "b/" in result["overview"]
        assert "c/" in result["overview"]
        assert "deep.py" not in result["overview"]

    async def test_custom_depth_is_clamped(self, tmp_path: Path) -> None:
        from horizon.tools.repo_overview import repo_overview

        _write(tmp_path / "a" / "b" / "file.py", "x\n")

        result = await repo_overview(path=str(tmp_path), depth=1)

        assert "a/" in result["overview"]
        assert "b/" not in result["overview"]

    async def test_ignores_noise_dirs(self, tmp_path: Path) -> None:
        from horizon.tools.repo_overview import repo_overview

        _write(tmp_path / ".venv" / "lib" / "junk.py", "x\n")
        _write(tmp_path / "node_modules" / "pkg" / "index.js", "x\n")
        _write(tmp_path / "__pycache__" / "m.pyc", "x\n")
        _write(tmp_path / "src" / "real.py", "x\n")

        result = await repo_overview(path=str(tmp_path))

        assert "src/" in result["overview"]
        assert "real.py" in result["overview"]
        assert ".venv" not in result["overview"]
        assert "node_modules" not in result["overview"]
        assert "__pycache__" not in result["overview"]

    async def test_tree_is_capped_and_flags_truncated(
        self, tmp_path: Path
    ) -> None:
        from horizon.tools.repo_overview import _STRUCTURE_LIMIT, repo_overview

        for i in range(_STRUCTURE_LIMIT + 50):
            _write(tmp_path / f"file_{i:04d}.py", "x\n")

        result = await repo_overview(path=str(tmp_path))

        tree_lines = [
            ln for ln in result["overview"].splitlines() if ln.endswith(".py")
        ]
        assert len(tree_lines) <= _STRUCTURE_LIMIT
        assert result["truncated"] is True


class TestErrors:
    async def test_missing_path_returns_error(self, tmp_path: Path) -> None:
        from horizon.tools.repo_overview import repo_overview

        result = await repo_overview(path=str(tmp_path / "nope"))

        assert result["success"] is False
        assert result["error"]

    async def test_path_outside_root_rejected(self, tmp_path: Path) -> None:
        from horizon.tools.repo_overview import repo_overview

        result = await repo_overview(path="../../etc")

        assert result["success"] is False
        assert result["error"]


class TestModuleSurface:
    def test_exported_from_package(self) -> None:
        from horizon.tools import repo_overview as exported
        from horizon.tools.repo_overview import repo_overview

        assert exported is repo_overview

    def test_registered_on_root_agent(self) -> None:
        from horizon.agent import root_agent

        names = {
            getattr(t, "__name__", getattr(t, "name", ""))
            for t in root_agent.tools
        }
        assert "repo_overview" in names
