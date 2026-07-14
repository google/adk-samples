"""
Generate a lightweight `tests/test_runnability.py` for a Python recipe.

What the generated test does — nothing more:
  1. Import the recipe's agent module.
  2. Assert `root_agent is not None`.
  3. If the module also defines a top-level `app`, assert it's non-None too.

The interesting work is generating the *right* test for THIS recipe. The
recipe's agent.py often has import-time side effects (`google.auth.default()`,
`vertexai.init()`) that would crash a naive `import`. We parse agent.py with
`ast` to figure out which mocks/env vars are actually needed and only include
those — no dead safety belt for recipes that don't need it, no missing mock
for recipes that do.

Detection scope:
  - `agent.py` itself: top-level assignments (root_agent, app), top-level
    calls (vertexai.init, google.auth.default), env var reads.
  - Sibling `.py` files in the same package directory: only for env-var
    convention scan (e.g. INTEGRATION_TEST, used by helpers that agent.py
    imports at module-load time).

Usage:

    python generate_runnability_test.py --recipe-dir <RECIPE_DIR> --dry-run
    python generate_runnability_test.py --recipe-dir <RECIPE_DIR>
    python generate_runnability_test.py --recipe-dir <RECIPE_DIR> --overwrite
    python generate_runnability_test.py --recipe-dir <RECIPE_DIR> \\
        --agent-file path/to/entry.py

Output: JSON report on stdout, structured for a coding agent to render.

Exit codes:
  0  dry-run: always. apply: file written (or was already correct).
  1  apply mode refused to overwrite an existing file (re-run with
     --overwrite to accept clobbering).
  2  hard error: recipe-dir invalid, agent.py not found, or parse failure.
"""

import argparse
import ast
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------- Constants ------------------------------------------------------

# Directories the recipe walker must never descend into.
#
# Two reasons for exclusion:
#   1. Virtualenvs, build artefacts, and tool caches contain third-party
#      code whose side-effects (INTEGRATION_TEST reads, vertexai calls) are
#      not the recipe's runtime code path — scanning them produces noise.
#   2. `tests/` is excluded so the generator does not read its own output
#      (a hand-written test that sets `INTEGRATION_TEST` via
#      `os.environ["INTEGRATION_TEST"] = "TRUE"` would otherwise be seen as
#      a recipe-source signal and self-perpetuate the env-var injection).
#      It also prevents a `tests/agent.py` test double from being picked
#      as the entry point instead of the real `app/agent.py`.
#
# Dot-directories (`.venv`, `.git`, `.tox`, `.pytest_cache`, `.ruff_cache`,
# `.mypy_cache`, `.idea`, `.vscode`, …) are pruned by a separate rule in
# `_walk_recipe_pyfiles`. `*.egg-info` suffix pruning is likewise there.
IGNORED_DIRS_NON_DOT = frozenset(
    {
        "venv",
        "build",
        "dist",
        "__pycache__",
        "node_modules",
        "tests",
    }
)

LICENSE_HEADER = """\
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
"""


# ---------- Report dataclasses ---------------------------------------------


@dataclass
class Detections:
    has_root_agent: bool = False
    has_app: bool = False
    needs_vertexai_patch: bool = False
    needs_gcp_project_env: bool = False
    needs_integration_test_env: bool = False


@dataclass
class Report:
    recipe_dir: str
    mode: str  # "dry-run" or "apply"
    agent_file: str | None = None
    module_name: str | None = None
    target_path: str | None = None
    detections: Detections = field(default_factory=Detections)
    test_content: str | None = None
    action: str | None = None  # would_write / wrote / refused_overwrite / error
    message: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "recipe_dir": self.recipe_dir,
                "mode": self.mode,
                "agent_file": self.agent_file,
                "module_name": self.module_name,
                "target_path": self.target_path,
                "detections": asdict(self.detections),
                "test_content": self.test_content,
                "action": self.action,
                "message": self.message,
            },
            indent=2,
        )


# ---------- agent.py discovery ---------------------------------------------


def _walk_recipe_pyfiles(recipe_dir: Path) -> list[Path]:
    """Yield every .py file under recipe_dir, safely pruning venv/build/etc."""
    found: list[Path] = []
    for root, dirs, files in os.walk(recipe_dir):
        dirs[:] = [
            d
            for d in dirs
            if d not in IGNORED_DIRS_NON_DOT
            and not d.startswith(".")
            and not d.endswith(".egg-info")
        ]
        for name in files:
            if name.endswith(".py"):
                found.append(Path(root) / name)
    return sorted(found)


def find_agent_file(recipe_dir: Path, override: Path | None) -> Path | None:
    """Return the agent.py to generate a test for.

    If override is given, resolve it (relative to recipe_dir if not absolute).
    Otherwise, prefer the shallowest agent.py under recipe_dir.
    """
    if override:
        path = override if override.is_absolute() else recipe_dir / override
        return path if path.is_file() else None

    candidates = []
    for p in _walk_recipe_pyfiles(recipe_dir):
        if p.name == "agent.py":
            depth = len(p.relative_to(recipe_dir).parts)
            candidates.append((depth, str(p), p))
    if not candidates:
        return None
    # Shallowest first, then alphabetical for a deterministic tie-break.
    candidates.sort()
    return candidates[0][2]


def module_path_from_file(agent_file: Path, recipe_dir: Path) -> str:
    """Convert `<recipe>/app/agent.py` to `app.agent`."""
    rel = agent_file.relative_to(recipe_dir)
    parts = list(rel.parts)
    parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


# ---------- AST detection helpers ------------------------------------------


def _call_target(call: ast.Call) -> str | None:
    """Dotted attribute chain for a Call's callee. `vertexai.init` etc."""
    parts: list[str] = []
    node: Any = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _collect_target_names(target: ast.expr) -> set[str]:
    """Return every `Name` that this assignment target binds.

    Handles the four forms the Python grammar allows on the left of `=`:
      - `x = ...`                          -> ast.Name
      - `x, y = ...` / `[x, y] = ...`      -> ast.Tuple / ast.List
      - `*rest = ...`                      -> ast.Starred
      - `obj.attr = ...` / `obj[key] = ...` -> ast.Attribute / ast.Subscript
        (ignored — these don't bind a module-level name)
    """
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names |= _collect_target_names(elt)
    elif isinstance(target, ast.Starred):
        names |= _collect_target_names(target.value)
    # Attribute / Subscript targets don't create module-level names — skip.
    return names


def find_top_level_assignments(module: ast.Module) -> set[str]:
    """Names bound at module top level by Assign or AnnAssign.

    Handles tuple / list / starred unpacking so patterns like
    `root_agent, app = _build()` still surface both names.
    """
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names |= _collect_target_names(target)
        elif isinstance(node, ast.AnnAssign):
            # PEP 526: AnnAssign has a single simple target — no unpacking
            # allowed at the grammar level, but the target may still be a
            # non-Name (Attribute/Subscript). Reuse the same helper.
            names |= _collect_target_names(node.target)
    return names


def find_call_targets(module: ast.Module) -> set[str]:
    """Dotted names of every Call anywhere in the module (any depth)."""
    names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            n = _call_target(node)
            if n:
                names.add(n)
    return names


def find_env_var_reads(module: ast.Module) -> set[str]:
    """Env var names read via os.getenv/os.environ.get/os.environ[...]."""
    vars: set[str] = set()
    for node in ast.walk(module):
        # os.getenv("VAR", ...) / os.environ.get("VAR", ...)
        if isinstance(node, ast.Call) and node.args:
            target = _call_target(node)
            if target in ("os.getenv", "os.environ.get"):
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(
                    first.value, str
                ):
                    vars.add(first.value)
        # os.environ["VAR"]  (read or write side; either counts)
        if isinstance(node, ast.Subscript):
            base = node.value
            if (
                isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name)
                and base.value.id == "os"
                and base.attr == "environ"
            ):
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    vars.add(sl.value)
    return vars


def find_imported_modules(module: ast.Module) -> set[str]:
    """Return the dotted module names referenced by any import statement.

    Both `Import` and `ImportFrom` forms are covered so aliased imports
    (`from vertexai import init`, `import vertexai as vai`) still contribute
    their canonical module name. Used as a fallback signal for detectors
    that would otherwise miss non-standard import styles: e.g. a call site
    `init()` after `from vertexai import init` is parsed as a bare `Name`
    and cannot be resolved by dotted-path matching alone.

    Values are the dotted module names — NOT the local aliases.
    """
    modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import a.b.c` -> "a.b.c"; `import a.b.c as x` -> "a.b.c"
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # `from a.b import x` -> "a.b"; also record the fully qualified
            # `a.b.x` so a caller can match either the parent or the leaf.
            modules.add(node.module)
            for alias in node.names:
                if alias.name != "*":
                    modules.add(f"{node.module}.{alias.name}")
    return modules


def _module_referenced(modules: set[str], prefix: str) -> bool:
    """True if any imported module equals `prefix` or starts with `prefix.`.

    Example: `_module_referenced({"vertexai.preview"}, "vertexai")` → True.
    """
    return prefix in modules or any(m.startswith(f"{prefix}.") for m in modules)


def detect_features(agent_file: Path, recipe_dir: Path) -> Detections:
    """Run every AST/scan detector against agent.py (and its sibling .py's
    for the INTEGRATION_TEST convention) and return the flags the generator
    needs to emit the right test."""
    module = ast.parse(
        agent_file.read_text(),
        filename=str(agent_file),
    )

    assignments = find_top_level_assignments(module)
    calls = find_call_targets(module)
    env_reads = find_env_var_reads(module)
    imports = find_imported_modules(module)

    d = Detections()
    d.has_root_agent = "root_agent" in assignments
    d.has_app = "app" in assignments

    # Primary signal: `vertexai.init` shows up in call targets. Fallback:
    # any import of a vertexai module — catches `from vertexai import init`
    # (where the call site is a bare `Name` and cannot be resolved by
    # dotted-path matching). Same idea for google.auth. False positives
    # (importing vertexai but never calling init) are cheap: a no-op patch.
    d.needs_vertexai_patch = "vertexai.init" in calls or _module_referenced(
        imports, "vertexai"
    )
    d.needs_gcp_project_env = (
        "google.auth.default" in calls
        or "GOOGLE_CLOUD_PROJECT" in env_reads
        or _module_referenced(imports, "google.auth")
    )
    # INTEGRATION_TEST is a per-package convention (agent.py imports helpers
    # like retrievers.py, and THOSE read the env var at module-load time via
    # top-level function calls). Scan every recipe .py file for it, not just
    # agent.py — otherwise we miss it for rag-*-style recipes. tests/ is
    # excluded by IGNORED_DIRS_NON_DOT so hand-written test files that set
    # INTEGRATION_TEST themselves don't cause the detector to self-perpetuate.
    d.needs_integration_test_env = _any_pyfile_reads_env_var(
        recipe_dir, "INTEGRATION_TEST"
    )
    return d


def _any_pyfile_reads_env_var(recipe_dir: Path, var: str) -> bool:
    """True if any .py file under recipe_dir (safely walked) reads env var
    `var` via os.getenv / os.environ.get / os.environ[...]."""
    for p in _walk_recipe_pyfiles(recipe_dir):
        try:
            module = ast.parse(p.read_text(), filename=str(p))
        except (SyntaxError, UnicodeDecodeError):
            continue
        if var in find_env_var_reads(module):
            return True
    return False


# ---------- Test-file generation -------------------------------------------


def generate_test_source(
    module_name: str,
    detections: Detections,
) -> str:
    """Emit the test_runnability.py contents for THIS recipe.

    Chooses between two shapes:
      * Minimal:  module-level `import <module>` + assertions.
      * Guarded:  env-var setup + optional `with patch(...)` around the
                  import, done inside the test function.
    """
    needs_env = (
        detections.needs_gcp_project_env
        or detections.needs_integration_test_env
    )
    has_side_effects = needs_env or detections.needs_vertexai_patch

    if not has_side_effects:
        return _emit_minimal(module_name, detections)
    return _emit_guarded(module_name, detections)


def _emit_minimal(module_name: str, detections: Detections) -> str:
    """Simplest test: import the module, assert the globals."""
    lines = [
        LICENSE_HEADER.rstrip(),
        '"""Runnability tests for the recipe."""',
        "",
        f"import {module_name}",
        "",
        "",
        "def test_agent_runnability() -> None:",
        '    """Verify agent.py imports and defines the expected globals."""',
        f"    assert {module_name}.root_agent is not None",
    ]
    if detections.has_app:
        lines.append(f"    assert {module_name}.app is not None")
    return "\n".join(lines) + "\n"


def _emit_guarded(module_name: str, detections: Detections) -> str:
    """Full-safety-belt test: env-var setup + optional vertexai patch.

    Import happens inside a `with patch(...)` context if vertexai.init is
    called at module load — the patch is only needed while the module loads.
    Assertions run AFTER the with block: at that point the module's globals
    are already stable, so keeping the patch active there would be misleading
    about what actually needs mocking.
    """
    # ---- imports ----
    imports = ["import os"]
    if detections.needs_vertexai_patch:
        imports.append("from unittest.mock import patch")

    # ---- explanatory comment (matches the hand-written recipe tests) ----
    comment_lines = _build_setup_comment(detections)

    # ---- env-var setup lines ----
    env_lines: list[str] = []
    if detections.needs_gcp_project_env:
        env_lines.append(
            '    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")'
        )
    if detections.needs_integration_test_env:
        env_lines.append(
            '    os.environ.setdefault("INTEGRATION_TEST", "TRUE")'
        )

    # ---- import block (with-patch context OR plain function-level import) --
    if detections.needs_vertexai_patch:
        import_lines = [
            '    with patch("vertexai.init"):',
            f"        import {module_name}",
        ]
    else:
        import_lines = [f"    import {module_name}"]

    # ---- assertions (always at function body level, not inside with) -----
    assertion_lines = [f"    assert {module_name}.root_agent is not None"]
    if detections.has_app:
        assertion_lines.append(f"    assert {module_name}.app is not None")

    # ---- assemble ----
    lines: list[str] = [
        LICENSE_HEADER.rstrip(),
        '"""Runnability tests for the recipe."""',
        "",
        *imports,
        "",
        "",
        "def test_agent_runnability() -> None:",
        '    """Verify agent.py imports and defines the expected globals."""',
        *comment_lines,
        *env_lines,
        "",
        *import_lines,
        "",
        *assertion_lines,
    ]
    return "\n".join(lines) + "\n"


def _build_setup_comment(detections: Detections) -> list[str]:
    """Return the `# ...` comment lines that explain why env-var setup and
    the vertexai patch are needed. Each returned line already includes its
    leading 4-space indent and `# ` marker, wrapped to fit under 80 chars.
    """
    bits: list[str] = []
    if detections.needs_gcp_project_env:
        bits.append(
            "Provide a dummy GCP project so agent.py does not call "
            "google.auth.default()"
        )
    if detections.needs_vertexai_patch:
        bits.append("mock vertexai.init to avoid a real GCP call")
    if not bits and detections.needs_integration_test_env:
        bits.append(
            "Set INTEGRATION_TEST so helpers imported by agent.py take "
            "their mock path"
        )
    prose = ", and ".join(bits) if bits else "Import-time setup"

    import textwrap

    wrapped = textwrap.wrap(
        f"{prose} — the setup must happen before the import.",
        width=76,
        break_long_words=False,
    )
    return [f"    # {line}" for line in wrapped]


# ---------- Orchestration --------------------------------------------------


def _analyze(
    recipe_dir: Path, agent_file_override: Path | None, report: Report
) -> bool:
    """Populate report.{agent_file, module_name, detections}.

    Returns True on success, False if a hard error was written to the report
    (in which case the caller should return the report as-is).
    """
    agent_file = find_agent_file(recipe_dir, agent_file_override)
    if agent_file is None:
        report.action = "error"
        report.message = (
            f"No agent.py found under {recipe_dir}. If your recipe uses a "
            f"different entry-point file, pass --agent-file <path> "
            f"(relative to the recipe dir or absolute)."
        )
        return False
    report.agent_file = str(agent_file)

    try:
        report.module_name = module_path_from_file(agent_file, recipe_dir)
    except ValueError as e:
        report.action = "error"
        report.message = (
            f"agent-file {agent_file} is not inside recipe-dir "
            f"{recipe_dir}: {e}"
        )
        return False

    try:
        report.detections = detect_features(agent_file, recipe_dir)
    except SyntaxError as e:
        report.action = "error"
        report.message = f"Failed to parse {agent_file}: {e}"
        return False

    if not report.detections.has_root_agent:
        report.message = (
            f"{agent_file} does not appear to have a top-level "
            f"`root_agent = ...` assignment; the generated test will fail "
            f"unless the module defines one when imported."
        )
    return True


def run(
    recipe_dir: Path,
    agent_file_override: Path | None,
    dry_run: bool,
    overwrite: bool,
) -> Report:
    mode = "dry-run" if dry_run else "apply"
    report = Report(recipe_dir=str(recipe_dir), mode=mode)

    if not _analyze(recipe_dir, agent_file_override, report):
        return report

    report.test_content = generate_test_source(
        report.module_name, report.detections
    )
    tests_dir = recipe_dir / "tests"
    target = tests_dir / "test_runnability.py"
    report.target_path = str(target)

    if dry_run:
        report.action = "would_write"
        return report

    if target.exists() and not overwrite:
        report.action = "refused_overwrite"
        report.message = (
            f"{target} already exists. Re-run with --overwrite to accept "
            f"clobbering."
        )
        return report

    try:
        tests_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(report.test_content)
    except OSError as e:
        report.action = "error"
        report.message = f"Failed to write {target}: {e}"
        return report

    report.action = "wrote"
    report.message = f"Wrote {target}."
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Generate tests/test_runnability.py for a Python recipe."),
    )
    parser.add_argument(
        "--recipe-dir",
        required=True,
        type=Path,
        help="Path to the recipe root (e.g. core/python/foo).",
    )
    parser.add_argument(
        "--agent-file",
        type=Path,
        default=None,
        help=(
            "Override auto-detection of the entry-point file "
            "(relative to --recipe-dir or absolute). "
            "Default: shallowest agent.py under --recipe-dir."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the JSON report (including generated content) without "
        "writing any file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing tests/test_runnability.py. Default: "
        "refuse and exit 1.",
    )
    args = parser.parse_args()

    if not args.recipe_dir.is_dir():
        print(
            f"Error: --recipe-dir {args.recipe_dir} is not a directory.",
            file=sys.stderr,
        )
        return 2

    try:
        report = run(
            recipe_dir=args.recipe_dir,
            agent_file_override=args.agent_file,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except Exception as e:  # final safety net for the CLI
        report = Report(
            recipe_dir=str(args.recipe_dir),
            mode="dry-run" if args.dry_run else "apply",
        )
        report.action = "error"
        report.message = (
            f"Unhandled {type(e).__name__}: {e}. This is a bug in the "
            f"generate-python-runnability-test skill; please report it."
        )

    print(report.to_json())

    if report.action == "error":
        return 2
    if report.action == "refused_overwrite":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
