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

"""repo_overview: one-call orientation for an unfamiliar workspace.

Ships an embedded Python walker run via ``env.execute`` — the same
portable pattern as ``file_ops.search_files`` — so it works against both
``LocalEnvironment`` and the remote ``SandboxEnvironment`` (where the
workspace is on a different host and only ``execute`` reaches it).
"""

from __future__ import annotations

import json
import shlex
from typing import Any

from horizon.environment_context import active_environment
from horizon.tools.file_ops import _error, _state_of, extract_delimited
from horizon.workspace_window import resolve_in_window, window_dirs

_OVERVIEW_DELIMITER = "###__LHA_REPO_OVERVIEW__###"
_STRUCTURE_LIMIT = 200
_DEFAULT_DEPTH = 3
_MIN_DEPTH = 1
_MAX_DEPTH = 6

_IGNORED_DIRS = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".next",
    "target",
    "vendor",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".idea",
    ".tox",
]

# Order matters: Python first (workspaces here are usually Python).
_ECOSYSTEM_MARKERS = [
    ("Python", ("pyproject.toml", "requirements.txt", "setup.py", "Pipfile")),
    ("Node.js", ("package.json",)),
    ("Go", ("go.mod",)),
    ("Rust", ("Cargo.toml",)),
    ("Ruby", ("Gemfile",)),
    ("Java/Kotlin", ("build.gradle", "build.gradle.kts", "pom.xml")),
    ("PHP", ("composer.json",)),
]

_DEPENDENCY_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "Pipfile.lock",
    "uv.lock",
    "poetry.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lock",
    "bun.lockb",
    "go.mod",
    "go.sum",
    "Cargo.toml",
    "Cargo.lock",
    "Gemfile",
    "Gemfile.lock",
    "build.gradle",
    "build.gradle.kts",
    "pom.xml",
    "composer.json",
]


def _build_overview_script(root: str, depth: int) -> str:
    return (
        "import json, os, sys\n"
        "try:\n"
        "    import tomllib as _toml\n"
        "except ModuleNotFoundError:\n"
        "    try:\n"
        "        import tomli as _toml\n"
        "    except ModuleNotFoundError:\n"
        "        _toml = None\n"
        f"_root = {root!r}\n"
        f"_depth = {depth!r}\n"
        f"_limit = {_STRUCTURE_LIMIT!r}\n"
        f"_ignored = set({_IGNORED_DIRS!r})\n"
        f"_eco_markers = {_ECOSYSTEM_MARKERS!r}\n"
        f"_dep_files = {_DEPENDENCY_FILES!r}\n"
        "if not os.path.isdir(_root):\n"
        "    sys.stderr.write('not a directory: ' + _root + '\\n')\n"
        "    sys.exit(2)\n"
        "try:\n"
        "    _top = set(os.listdir(_root))\n"
        "except OSError as _e:\n"
        "    sys.stderr.write(str(_e) + '\\n'); sys.exit(2)\n"
        "_ecosystems = [n for (n, ms) in _eco_markers if any(m in _top for m in ms)]\n"
        "_dependency_files = [f for f in _dep_files if f in _top]\n"
        "_pkg_mgrs = []\n"
        "if 'uv.lock' in _top: _pkg_mgrs.append('uv')\n"
        "if 'poetry.lock' in _top: _pkg_mgrs.append('poetry')\n"
        "if 'Pipfile.lock' in _top or 'Pipfile' in _top: _pkg_mgrs.append('pipenv')\n"
        "if 'bun.lock' in _top or 'bun.lockb' in _top: _pkg_mgrs.append('bun')\n"
        "if 'pnpm-lock.yaml' in _top: _pkg_mgrs.append('pnpm')\n"
        "if 'yarn.lock' in _top: _pkg_mgrs.append('yarn')\n"
        "if 'package-lock.json' in _top: _pkg_mgrs.append('npm')\n"
        "_entrypoints = []\n"
        "def _add(label, value):\n"
        "    _entrypoints.append(label + ': ' + value)\n"
        "if 'pyproject.toml' in _top and _toml is not None:\n"
        "    try:\n"
        "        with open(os.path.join(_root, 'pyproject.toml'), 'rb') as _fh:\n"
        "            _pp = _toml.load(_fh)\n"
        "        _proj = _pp.get('project', {}) if isinstance(_pp, dict) else {}\n"
        "        for _sect in ('scripts', 'gui-scripts'):\n"
        "            _scripts = _proj.get(_sect, {})\n"
        "            if isinstance(_scripts, dict):\n"
        "                for _k in list(_scripts)[:20]:\n"
        "                    _add('script', _k + ' = ' + str(_scripts[_k]))\n"
        "    except Exception:\n"
        "        pass\n"
        "if 'package.json' in _top:\n"
        "    try:\n"
        "        _pjp = os.path.join(_root, 'package.json')\n"
        "        with open(_pjp, 'r', encoding='utf-8') as _fh:\n"
        "            _pj = json.load(_fh)\n"
        "        if isinstance(_pj, dict):\n"
        "            for _key in ('main', 'module', 'bin'):\n"
        "                _v = _pj.get(_key)\n"
        "                if isinstance(_v, str):\n"
        "                    _add(_key, _v)\n"
        "                elif isinstance(_v, dict):\n"
        "                    for _bk in list(_v)[:10]:\n"
        "                        _add(_key, _bk + ' -> ' + str(_v[_bk]))\n"
        "    except Exception:\n"
        "        pass\n"
        "_conv = ['__main__.py', 'main.py', 'app/agent.py', 'app/main.py',\n"
        "         'manage.py', 'app.py', 'src/main.py']\n"
        "for _rel in _conv:\n"
        "    if os.path.isfile(os.path.join(_root, _rel)):\n"
        "        _add('file', _rel)\n"
        "_srcdir = os.path.join(_root, 'src')\n"
        "if os.path.isdir(_srcdir):\n"
        "    try:\n"
        "        for _pkg in sorted(os.listdir(_srcdir)):\n"
        "            _mm = os.path.join(_srcdir, _pkg, '__main__.py')\n"
        "            if os.path.isfile(_mm):\n"
        "                _add('file', 'src/' + _pkg + '/__main__.py')\n"
        "    except OSError:\n"
        "        pass\n"
        "_lines = []\n"
        "_truncated = [False]\n"
        "def _walk(_d, _level):\n"
        "    if _level >= _depth or len(_lines) >= _limit:\n"
        "        if len(_lines) >= _limit:\n"
        "            _truncated[0] = True\n"
        "        return\n"
        "    try:\n"
        "        _names = os.listdir(_d)\n"
        "    except OSError:\n"
        "        return\n"
        "    _items = []\n"
        "    for _n in _names:\n"
        "        if _n in _ignored or _n.startswith('.'):\n"
        "            continue\n"
        "        _full = os.path.join(_d, _n)\n"
        "        _items.append((_n, _full, os.path.isdir(_full)))\n"
        "    _items.sort(key=lambda _it: (not _it[2], _it[0].lower()))\n"
        "    for (_n, _full, _isdir) in _items:\n"
        "        if len(_lines) >= _limit:\n"
        "            _truncated[0] = True\n"
        "            return\n"
        "        _lines.append('  ' * _level + _n + ('/' if _isdir else ''))\n"
        "        if _isdir:\n"
        "            _walk(_full, _level + 1)\n"
        "_walk(_root, 0)\n"
        "_payload = {\n"
        "    'ecosystems': _ecosystems,\n"
        "    'package_managers': _pkg_mgrs,\n"
        "    'dependency_files': _dependency_files,\n"
        "    'entrypoints': _entrypoints,\n"
        "    'tree': _lines,\n"
        "    'truncated': _truncated[0],\n"
        "}\n"
        f"print({_OVERVIEW_DELIMITER!r} + json.dumps(_payload) + "
        f"{_OVERVIEW_DELIMITER!r})\n"
    )


def _render(root: str, depth: int, data: dict[str, Any]) -> str:
    out: list[str] = [f"Workspace: {root}", f"Depth: {depth}"]
    if data["ecosystems"]:
        out.append("Ecosystems: " + ", ".join(data["ecosystems"]))
    if data["package_managers"]:
        out.append("Package managers: " + ", ".join(data["package_managers"]))
    if data["dependency_files"]:
        out.append("Dependency files: " + ", ".join(data["dependency_files"]))
    if data["entrypoints"]:
        out.append("Likely entrypoints:")
        out.extend(f"- {e}" for e in data["entrypoints"])
    out.append("Structure:")
    out.extend(data["tree"])
    if data["truncated"]:
        out.append("(structure truncated)")
    return "\n".join(out)


async def repo_overview(
    *,
    path: str = ".",
    depth: int = _DEFAULT_DEPTH,
    scope: str = "window",
    tool_context: Any | None = None,
) -> dict[str, Any]:
    """Orient on a workspace in one call: ecosystems, dependency files, likely
    entrypoints, and a depth-limited structure tree. Use this first on an
    unfamiliar workspace instead of a series of `terminal ls` / `read_file`
    calls.

    `path` defaults to your current workspace focus (see `/workspace`); pass
    `scope="workspace"` (or `path="/"`) to survey the entire workspace across
    all projects. `depth` (1-6, default 3) caps tree nesting.
    """
    env_root = active_environment().working_dir.resolve()
    window = (
        [] if scope == "workspace" else window_dirs(_state_of(tool_context))
    )
    root, err = resolve_in_window(path, env_root, window)
    if err is not None:
        return _error(err)
    assert root is not None

    if not isinstance(depth, int) or depth < _MIN_DEPTH:
        depth = _DEFAULT_DEPTH
    depth = min(depth, _MAX_DEPTH)

    script = _build_overview_script(str(root), depth)
    command = f"python3 -c {shlex.quote(script)}"
    result = await active_environment().execute(command)
    if result.exit_code != 0:
        msg = result.stderr.strip() or "repo_overview failed"
        if "not a directory" in msg.lower() or "No such file" in msg:
            return _error(f"Workspace path invalid: {path}: {msg}")
        return _error(f"repo_overview failed for {path}: {msg}")

    payload = extract_delimited(result.stdout, _OVERVIEW_DELIMITER)
    if payload is None:
        return _error(
            f"repo_overview produced no result envelope for {path}; "
            f"stderr={result.stderr!r}"
        )
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        return _error(f"repo_overview produced invalid JSON: {exc}")

    return {
        "success": True,
        "overview": _render(str(root), depth, data),
        "ecosystems": data["ecosystems"],
        "package_managers": data["package_managers"],
        "dependency_files": data["dependency_files"],
        "entrypoints": data["entrypoints"],
        "depth": depth,
        "truncated": data["truncated"],
    }
