#!/usr/bin/env python3
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
extract_env_vars.py

Scans a Python recipe directory and:
  1. Detects all environment variable reads in non-test Python files.
  2. Creates or updates .env.example with any missing variables.
  3. Injects the load_dotenv() bootstrap snippet into the package __init__.py.
  4. Ensures python-dotenv>=1.0.0 is listed in pyproject.toml dependencies.
  5. Detects hardcoded model name strings, replaces them with
     os.getenv("MODEL_NAME"), and adds MODEL_NAME to .env.example.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

import tomllib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLACEHOLDER = "<TODO: update-this-value>"

# Model name prefixes that indicate a hardcoded model identifier.
# Kept in sync with validate-python-recipe.yml.
MODEL_PREFIXES: tuple[str, ...] = (
    "gemini-",
    "gemini-exp-",
    "imagen-",
    "claude-",
    "llama-",
    "meta/llama-",
    "mistral-",
    "codestral-",
    "phi-",
    "grok-",
    "command-",
    "jamba-",
)

LOAD_DOTENV_IMPORT = "from dotenv import load_dotenv"

LOAD_DOTENV_SNIPPET = """\
# Load variables from .env if present. In production the environment is
# already populated by the platform (Cloud Run, GKE, etc.), so a missing
# .env is expected and not an error.
load_dotenv()"""


# ---------------------------------------------------------------------------
# Step 1: Find Python files
# ---------------------------------------------------------------------------

# Directories that never contain first-party recipe source and MUST be skipped
# during scanning. Virtualenvs are the biggest hazard: a local `.venv/`
# containing installed third-party packages would otherwise pollute
# `.env.example` with unrelated env vars and match hundreds of hardcoded model
# strings inside third-party code.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        # Test directories (not part of the recipe's runtime code path).
        "tests",
        # Python virtualenvs (all common names).
        ".venv",
        "venv",
        "env",
        # Bytecode + tool caches.
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".eggs",
        # Build artifacts.
        "build",
        "dist",
        # VCS metadata.
        ".git",
        ".hg",
        ".svn",
        # JS toolchains (uncommon here but cheap to skip).
        "node_modules",
    }
)


def find_python_files(recipe_dir: Path) -> list[Path]:
    """
    Return all .py files under recipe_dir, skipping directories that are not
    part of the recipe's own source (see SKIP_DIRS).
    """
    return [
        p
        for p in sorted(recipe_dir.rglob("*.py"))
        if not SKIP_DIRS.intersection(p.relative_to(recipe_dir).parts)
    ]


# ---------------------------------------------------------------------------
# Step 2: Extract environment variable reads via AST
# ---------------------------------------------------------------------------


def _extract_var_from_node(
    node: ast.AST,
) -> tuple[str | None, str | None]:
    """
    Return (var_name, default) if node is an env-var read, else (None, None).

    Handles: os.getenv(), os.environ.get(), os.environ[].
    """

    def _str_const(n: ast.expr) -> str | None:
        return (
            n.value
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            else None
        )

    # os.getenv("VAR") / os.getenv("VAR", "default")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "getenv"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.args
    ):
        var_name = _str_const(node.args[0])
        default = _str_const(node.args[1]) if len(node.args) > 1 else None
        return var_name, default

    # os.environ.get("VAR") / os.environ.get("VAR", "default")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "environ"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "os"
        and node.args
    ):
        var_name = _str_const(node.args[0])
        default = _str_const(node.args[1]) if len(node.args) > 1 else None
        return var_name, default

    # os.environ["VAR"]
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "environ"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
    ):
        return _str_const(node.slice), None

    return None, None


def extract_env_vars(py_files: list[Path]) -> dict[str, str | None]:
    """
    Walk each file's AST and collect env var names + optional inline defaults.

    Returns:
      {VAR_NAME: default_value_or_None}
      When a variable appears multiple times, a non-None default wins.
    """
    found: dict[str, str | None] = {}

    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            print(
                f"[WARN] Could not parse {py_file} — skipping.", file=sys.stderr
            )
            continue

        for node in ast.walk(tree):
            var_name, default = _extract_var_from_node(node)
            if var_name and re.match(r"^[A-Z_][A-Z0-9_]*$", var_name):
                if var_name not in found or found[var_name] is None:
                    found[var_name] = default

    return found


# ---------------------------------------------------------------------------
# Step 3: Create / update .env.example
# ---------------------------------------------------------------------------


def read_defined_vars(env_example: Path) -> set[str]:
    """Return the set of variable names already declared in .env.example."""
    if not env_example.exists():
        return set()

    defined: set[str] = set()
    for line in env_example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Strip optional 'export ' prefix
        stripped = re.sub(r"^export\s+", "", stripped)
        m = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=", stripped)
        if m:
            defined.add(m.group(1))
    return defined


def update_env_example(
    env_example: Path,
    env_vars: dict[str, str | None],
    dry_run: bool = False,
) -> list[str]:
    """
    Append variables not yet in .env.example.
    Returns the list of variable names that were (or would be) added.

    When dry_run is True, no file is written; the return value still reports
    which variables would be added.
    """
    existing = read_defined_vars(env_example)
    to_add = {k: v for k, v in env_vars.items() if k not in existing}

    if not to_add:
        return []

    if dry_run:
        return sorted(to_add.keys())

    if env_example.exists():
        current = env_example.read_text(encoding="utf-8")
        if not current.endswith("\n"):
            current += "\n"
    else:
        current = ""

    block = (
        "\n# Environment variables extracted by"
        " extract-python-environment-variables\n"
    )
    for var in sorted(to_add):
        value = to_add[var] if to_add[var] is not None else PLACEHOLDER
        block += f"{var}={value}\n"

    env_example.write_text(current + block, encoding="utf-8")
    return sorted(to_add.keys())


# ---------------------------------------------------------------------------
# Shared helpers: file structure analysis
# ---------------------------------------------------------------------------


def _post_header_index(lines: list[str]) -> int:
    """
    Return the line index after which new top-level code should be inserted.

    Skips (in order):
      1. Leading license / comment block and blank lines.
      2. An optional module-level docstring (single- or triple-quoted).

    This prevents imports from being injected before the module docstring,
    which would cause documentation tools to miss it.
    """
    i = 0
    n = len(lines)

    # Skip license header (comment lines and blank lines)
    while i < n and (lines[i].strip().startswith("#") or not lines[i].strip()):
        i += 1

    # Skip module docstring if present
    if i < n:
        stripped = lines[i].strip()
        for quote in ('"""', "'''"):
            if not stripped.startswith(quote):
                continue
            rest = stripped[len(quote) :]
            if rest.endswith(quote) and len(rest) >= len(quote):
                i += 1  # single-line docstring
            else:
                i += 1  # multi-line: scan for closing quotes
                while i < n and quote not in lines[i]:
                    i += 1
                i += 1  # include the line that contains the closing quotes
            break

    return i


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """
    Return the id() of every ast.Constant that is a docstring.

    A docstring is the first statement of a module, class, or function body
    when that statement is a bare string expression.  Excluding these prevents
    the model-name replacement from corrupting documentation text.
    """
    ids: set[int] = set()

    def _mark(stmts: list[ast.stmt]) -> None:
        if (
            stmts
            and isinstance(stmts[0], ast.Expr)
            and isinstance(stmts[0].value, ast.Constant)
            and isinstance(stmts[0].value.value, str)
        ):
            ids.add(id(stmts[0].value))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef)):
            _mark(node.body)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _mark(node.body)

    return ids


def _flat_offset(lines: list[str], lineno: int, col: int) -> int:
    """Convert a 1-based lineno + 0-based col_offset to a flat char offset."""
    return sum(len(ln) for ln in lines[: lineno - 1]) + col


def _imports_os(tree: ast.AST) -> bool:
    """Return True if the module actually imports ``os`` (binds ``os``).

    A plain substring search for "import os" gives false positives when the
    text appears in a comment or docstring, so we inspect real import nodes.
    ``import os`` and ``import os.path`` both bind ``os``; ``import os as o``
    does not.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if name == "os":
                    return True
    return False


_RELATIVE_IMPORT_RE = re.compile(r"^\s*from\s+\.")
_NOQA_E402_SUFFIX = "  # noqa: E402 -- must come after load_dotenv()"


def _maybe_suppress_e402(line: str) -> str:
    """Append a `# noqa: E402` suffix to a relative-import line if absent.

    A no-op for anything that isn't a relative import (``from .x import ...``)
    or that already carries an E402 noqa comment.
    """
    if not _RELATIVE_IMPORT_RE.match(line):
        return line
    if "noqa" in line and "E402" in line:
        return line
    stripped = line.rstrip("\n")
    newline = line[len(stripped) :]  # preserve original line ending
    return stripped + _NOQA_E402_SUFFIX + newline


def _has_load_dotenv(tree: ast.AST) -> bool:
    """Return True if the module already imports load_dotenv / dotenv.

    AST-based so that a mention of "load_dotenv" in a docstring or comment does
    not suppress a legitimate injection.
    """
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "dotenv"
            and any(a.name == "load_dotenv" for a in node.names)
        ):
            return True
        if isinstance(node, ast.Import) and any(
            a.name == "dotenv" for a in node.names
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Step 4: Inject load_dotenv() into package __init__.py
# ---------------------------------------------------------------------------


def find_package_init(recipe_dir: Path) -> Path | None:
    """
    Return the __init__.py of the top-level Python package inside recipe_dir.

    Looks for immediate subdirectories that contain __init__.py, skipping
    test and hidden directories. Without this guard a ``tests/`` package that
    sorts before the real agent package would receive the load_dotenv()
    bootstrap by mistake.
    """
    for candidate in sorted(recipe_dir.iterdir()):
        if not candidate.is_dir():
            continue
        if candidate.name == "tests" or candidate.name.startswith("."):
            continue
        if (candidate / "__init__.py").exists():
            return candidate / "__init__.py"
    return None


def inject_load_dotenv(init_py: Path, dry_run: bool = False) -> bool:
    """
    Ensure load_dotenv import + bootstrap snippet exist in __init__.py.
    Returns True if the file was (or would be) modified.

    When dry_run is True, no file is written; the return value still reports
    whether the snippet would be injected.
    """
    content = init_py.read_text(encoding="utf-8")

    try:
        already_present = _has_load_dotenv(ast.parse(content))
    except SyntaxError:
        # Fall back to a conservative substring check on unparseable files.
        already_present = "load_dotenv" in content
    if already_present:
        return False  # Already present — nothing to do

    if dry_run:
        return True  # Would inject the bootstrap snippet.

    lines = content.splitlines(keepends=True)

    # Build the block to inject (import + blank line + snippet + blank line)
    inject_block = f"\n{LOAD_DOTENV_IMPORT}\n\n{LOAD_DOTENV_SNIPPET}\n"

    # Find the index of the last absolute import line so we can insert after it.
    # Relative imports (from .something) must come AFTER load_dotenv() so that
    # the env is populated before any package module-level code runs.
    last_import_idx: int = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_absolute_import = stripped.startswith("import ") or (
            stripped.startswith("from ") and not stripped.startswith("from .")
        )
        if is_absolute_import:
            last_import_idx = i

    if last_import_idx >= 0:
        lines.insert(last_import_idx + 1, inject_block)
    else:
        # No imports — insert after license header and any module docstring
        lines.insert(_post_header_index(lines), inject_block)

    # Any relative imports (`from .x import ...`) now sit AFTER the injected
    # load_dotenv() call, which would trigger Ruff E402 ("module-level import
    # not at top of file"). Suppress that warning per-line, since the ordering
    # is intentional — env must be populated before agent module-level code.
    lines = [_maybe_suppress_e402(ln) for ln in lines]

    init_py.write_text("".join(lines), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Step 5: Ensure python-dotenv in pyproject.toml
# ---------------------------------------------------------------------------


_PYPROJECT_HEADER_RE = re.compile(r"(?m)^\[project\][ \t]*(#.*)?$")
_NEXT_SECTION_RE = re.compile(r"(?m)^\[[^\]]+\]")
_DEPS_START_RE = re.compile(r"(?m)^\s*dependencies\s*=\s*\[")
_DEP_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


def _dependencies_has_python_dotenv(content: str) -> bool | None:
    """
    Report whether [project].dependencies already lists python-dotenv.

    Uses tomllib (stdlib since Python 3.11) so it correctly handles extras
    (`google-adk[gcp]>=2.0.0`), single-line vs. multi-line arrays, and any
    other TOML surface syntax — avoiding the false negatives the previous
    regex-based check produced when an earlier dep contained a `]`.

    Returns:
      True  — python-dotenv is present in [project].dependencies.
      False — [project].dependencies exists and does NOT include it.
      None  — [project] or [project].dependencies is missing, or the file
              is not valid TOML.
    """
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return None
    project = data.get("project") if isinstance(data, dict) else None
    if not isinstance(project, dict):
        return None
    deps = project.get("dependencies")
    if not isinstance(deps, list):
        return None
    for dep in deps:
        if not isinstance(dep, str):
            continue
        name = _DEP_NAME_RE.match(dep)
        if name and name.group(1).lower() == "python-dotenv":
            return True
    return False


def _scan_matching_close_bracket(content: str, open_idx: int) -> int | None:
    """
    Given the offset of a '[' in `content`, return the offset of its
    matching ']'. Depth-aware (skips nested brackets) and quote-aware
    (skips brackets inside string literals). Returns None if not matched.
    """
    depth = 0
    in_str: str | None = None  # None, or the currently-open quote char.
    i = open_idx
    while i < len(content):
        c = content[i]
        if in_str is not None:
            if c == "\\":
                i += 2  # Skip the escaped character.
                continue
            if c == in_str:
                in_str = None
        elif c in ('"', "'"):
            in_str = c
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _find_dependencies_close_bracket(content: str) -> int | None:
    """
    Return the character offset of the closing ']' of the top-level
    [project].dependencies array, or None if the array is not present.

    Extras brackets (`google-adk[gcp]`) and any `]` inside a dep string
    are correctly ignored — see `_scan_matching_close_bracket`.
    """
    proj = _PYPROJECT_HEADER_RE.search(content)
    if not proj:
        return None
    # Bound the [project] table: up to the next top-level section header.
    remainder = content[proj.end() :]
    next_sec = _NEXT_SECTION_RE.search(remainder)
    proj_end = proj.end() + (next_sec.start() if next_sec else len(remainder))
    deps_start = _DEPS_START_RE.search(content, proj.end(), proj_end)
    if not deps_start:
        return None
    # Position of '[' is deps_start.end() - 1 (regex ends just past it).
    return _scan_matching_close_bracket(content, deps_start.end() - 1)


def _insert_before_close(content: str, close_idx: int) -> str:
    """
    Insert a `"python-dotenv>=1.0.0",` line into the dependencies array
    immediately before the closing ']' at `close_idx`.

    Ensures the preceding entry ends with a comma. The insertion is always
    placed on its own line with a 4-space indent — valid TOML in every case,
    and it stays readable whether the source array was multi-line or
    single-line.
    """
    prefix = content[:close_idx]
    suffix = content[close_idx:]

    # Trim trailing whitespace between the last entry and ']' so we control
    # the layout of the insertion.
    trimmed = prefix.rstrip()

    # If the array has any content, make sure the last entry has a trailing
    # comma before we append our own.
    array_open = trimmed.rfind("[")
    array_body = trimmed[array_open + 1 :] if array_open >= 0 else ""
    if array_body.strip() and not trimmed.endswith(","):
        trimmed += ","

    return trimmed + '\n    "python-dotenv>=1.0.0",\n' + suffix


def _compute_pyproject_with_dotenv(content: str) -> str | None:
    """
    Return `content` with `python-dotenv>=1.0.0` added to
    `[project].dependencies`, or None if nothing should be written
    (already present, no `[project]` table, or no safe place to insert).
    """
    status = _dependencies_has_python_dotenv(content)
    if status is True:
        return None  # Already present — nothing to do.
    if status is False:
        # [project].dependencies exists but is missing python-dotenv.
        close_idx = _find_dependencies_close_bracket(content)
        if close_idx is None:
            return None  # Defensive: shouldn't happen given status.
        return _insert_before_close(content, close_idx)
    # status is None. Create the array if [project] exists, otherwise bail.
    project_header = _PYPROJECT_HEADER_RE.search(content)
    if not project_header:
        return None
    insert_at = project_header.end()
    block = '\ndependencies = [\n    "python-dotenv>=1.0.0",\n]'
    return content[:insert_at] + block + content[insert_at:]


def ensure_python_dotenv_dependency(
    pyproject: Path, dry_run: bool = False
) -> bool:
    """
    Add `python-dotenv>=1.0.0` to `[project].dependencies` if it is not
    already present there. Returns True if the file was (or would be)
    modified, False otherwise.

    Detection uses `tomllib` (parse the file, walk `[project].dependencies`)
    so extras like `google-adk[gcp]` and non-standard formatting don't
    produce false negatives — the root cause of the earlier bug that
    corrupted pyproject.toml files.

    Insertion is bracket-depth-aware and quote-aware. After building the new
    content, we round-trip it through `_dependencies_has_python_dotenv` and
    refuse to write anything that isn't valid TOML with python-dotenv now
    present under `[project].dependencies`.

    When `dry_run` is True, no file is written; the return value still
    reflects whether the dependency would be added.
    """
    if not pyproject.exists():
        return False
    content = pyproject.read_text(encoding="utf-8")
    new_content = _compute_pyproject_with_dotenv(content)
    if new_content is None:
        return False
    # Round-trip check: refuse to write output that is not valid TOML with
    # python-dotenv now under [project].dependencies. Hard backstop against
    # any future insertion bug producing invalid TOML.
    if _dependencies_has_python_dotenv(new_content) is not True:
        return False
    if not dry_run:
        pyproject.write_text(new_content, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Step 6: Detect and extract hardcoded model names
# ---------------------------------------------------------------------------


def extract_hardcoded_models(
    py_files: list[Path],
) -> dict[Path, list[tuple[int, str]]]:
    """
    Find string literals that look like hardcoded model names in Python files.

    Uses AST to walk string constants and checks whether the value starts with
    any known model prefix (same list as validate-python-recipe.yml).
    Docstring nodes are excluded to avoid false positives from documentation.

    Returns:
      {file_path: [(line_number, model_string), ...]}
    """
    hits: dict[Path, list[tuple[int, str]]] = {}

    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        docstring_ids = _docstring_node_ids(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if id(node) in docstring_ids:
                continue
            if not isinstance(node.value, str):
                continue
            if any(node.value.startswith(prefix) for prefix in MODEL_PREFIXES):
                hits.setdefault(py_file, []).append((node.lineno, node.value))

    return hits


def _model_str_to_suffix(model_str: str) -> str:
    """
    Derive a human-readable env var suffix from a model string.

    Strategy:
      1. Uppercase the model string.
      2. Replace any character that is not A-Z or 0-9 with an underscore.
      3. Collapse consecutive underscores into one.
      4. Strip leading/trailing underscores.

    Examples:
      "gemini-3.5-flash"      → "GEMINI_3_5_FLASH"
      "gemini-embedding-001"  → "GEMINI_EMBEDDING_001"
      "claude-3-sonnet"       → "CLAUDE_3_SONNET"
      "llama-3.1-70b"         → "LLAMA_3_1_70B"
    """
    suffix = re.sub(r"[^A-Z0-9]+", "_", model_str.upper())
    return suffix.strip("_")


def assign_model_var_names(model_strings: set[str]) -> dict[str, str]:
    """
    Assign a standardised MODEL_NAME_* env var name to each unique model string.

    Rules (applied to the sorted list for determinism):
      - If there is only one model → MODEL_NAME (no suffix).
      - Otherwise derive a suffix from the model string itself using
        _model_str_to_suffix().
        E.g., "gemini-3.5-flash" → MODEL_NAME_GEMINI_3_5_FLASH.
      - If two different model strings produce the same derived suffix
        (collision), append _2, _3, … to disambiguate.

    Returns:
      {model_string: env_var_name}
    """
    sorted_strings = sorted(model_strings)

    # Single model — plain MODEL_NAME, no suffix needed.
    if len(sorted_strings) == 1:
        return {sorted_strings[0]: "MODEL_NAME"}

    mapping: dict[str, str] = {}
    seen_suffixes: dict[str, int] = {}  # suffix → count of times used so far

    for model_str in sorted_strings:
        base_suffix = _model_str_to_suffix(model_str)
        count = seen_suffixes.get(base_suffix, 0)
        seen_suffixes[base_suffix] = count + 1
        if count == 0:
            var_name = f"MODEL_NAME_{base_suffix}"
        else:
            var_name = f"MODEL_NAME_{base_suffix}_{count + 1}"
        mapping[model_str] = var_name

    return mapping


def _model_replacement(
    node: ast.AST,
    docstring_ids: set[int],
    name_map: dict[str, str],
    lines: list[str],
) -> tuple[int, int, str, str, str] | None:
    """
    Return (start, end, new_text, model_str, var_name) if node is a
    replaceable hardcoded model string, else None.
    """
    if not isinstance(node, ast.Constant):
        return None
    if id(node) in docstring_ids:
        return None
    if not isinstance(node.value, str):
        return None
    var_name = name_map.get(node.value)
    if not var_name:
        return None
    start = _flat_offset(lines, node.lineno, node.col_offset)
    end = _flat_offset(lines, node.end_lineno, node.end_col_offset)
    return start, end, f'os.getenv("{var_name}")', node.value, var_name


def replace_hardcoded_models(
    py_files: list[Path],
    hits: dict[Path, list[tuple[int, str]]],
    name_map: dict[str, str],
    dry_run: bool = False,
) -> dict[str, str]:
    """
    Replace each hardcoded model string with the correct
    os.getenv("MODEL_NAME_*") call in-place, using the mapping produced by
    assign_model_var_names().

    Replacement is AST-position-based, which means:
      - All quote styles (single, double, triple, raw) are handled correctly
        because the AST abstracts away quoting entirely.
      - Only actual string-literal AST nodes are replaced — comments,
        docstrings, and f-string fragments are never touched.

    Also ensures `import os` is present in every modified file.

    When dry_run is True, no file is written; the returned mapping still
    reports which substitutions would be made.

    Returns a dict of {model_string: env_var_name} for the substitutions made.
    """
    substituted: dict[str, str] = {}

    for py_file in hits:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        docstring_ids = _docstring_node_ids(tree)
        lines = source.splitlines(keepends=True)

        # Collect (start_offset, end_offset, replacement_text) for each hit
        replacements: list[tuple[int, int, str]] = []
        for node in ast.walk(tree):
            replacement = _model_replacement(
                node, docstring_ids, name_map, lines
            )
            if replacement is None:
                continue
            start, end, new_text, model_str, var_name = replacement
            replacements.append((start, end, new_text))
            substituted[model_str] = var_name

        if not replacements:
            continue

        if dry_run:
            continue  # Detection recorded in `substituted`; skip writing.

        # Apply in reverse order so earlier offsets stay valid
        replacements.sort(key=lambda x: x[0], reverse=True)
        chars = list(source)
        for start, end, new_text in replacements:
            chars[start:end] = list(new_text)
        modified = "".join(chars)

        # Ensure `import os` is present, placed after license + docstring.
        # AST-based so "import os" appearing only in a comment or docstring
        # does not suppress the real import (which would leave the injected
        # os.getenv(...) calls raising NameError at runtime).
        # Include a trailing blank line so the import block is well-formatted.
        try:
            already_imports_os = _imports_os(ast.parse(modified))
        except SyntaxError:
            already_imports_os = "import os" in modified
        if not already_imports_os:
            mod_lines = modified.splitlines(keepends=True)
            idx = _post_header_index(mod_lines)
            # Avoid double blank lines if the line at idx is already blank
            suffix = (
                "\n" if idx < len(mod_lines) and mod_lines[idx].strip() else ""
            )
            mod_lines.insert(idx, f"import os\n{suffix}")
            modified = "".join(mod_lines)

        py_file.write_text(modified, encoding="utf-8")

    return substituted


# ---------------------------------------------------------------------------
# Main — step runners
# ---------------------------------------------------------------------------


def _tag(dry_run: bool) -> str:
    """Status prefix used in log lines: DRY-RUN when nothing is written."""
    return "DRY-RUN" if dry_run else "PASS"


def run_step_env_vars(
    recipe_dir: Path, py_files: list[Path], dry_run: bool = False
) -> tuple[Path, dict[str, str | None]]:
    """Steps 2 + 3: extract env var reads and update .env.example."""
    env_vars = extract_env_vars(py_files)
    if env_vars:
        print(f"\n[INFO] Detected {len(env_vars)} environment variable(s):")
        for var in sorted(env_vars):
            default = env_vars[var]
            suffix = f"  (default: {default!r})" if default is not None else ""
            print(f"       {var}{suffix}")
    else:
        print("\n[INFO] No environment variable reads detected.")

    env_example = recipe_dir / ".env.example"
    added = update_env_example(env_example, env_vars, dry_run=dry_run)
    if added:
        verb = "Would add" if dry_run else "Added"
        print(
            f"\n[{_tag(dry_run)}] {verb} {len(added)} variable(s) to "
            ".env.example: " + ", ".join(added)
        )
    else:
        print(
            "\n[PASS] .env.example is already up to date — no variables added."
        )

    return env_example, env_vars


def run_step_load_dotenv(recipe_dir: Path, dry_run: bool = False) -> None:
    """Step 4: inject load_dotenv() bootstrap into the package __init__.py."""
    init_py = find_package_init(recipe_dir)
    if not init_py:
        print(
            "[WARN] No Python package (subdirectory with __init__.py) found. "
            "load_dotenv() injection skipped."
        )
        return
    rel = init_py.relative_to(recipe_dir)
    if inject_load_dotenv(init_py, dry_run=dry_run):
        verb = "Would inject" if dry_run else "Injected"
        print(f"[{_tag(dry_run)}] {verb} load_dotenv() bootstrap into {rel}")
    else:
        print(f"[PASS] load_dotenv() already present in {rel} — skipped.")


def run_step_pyproject(recipe_dir: Path, dry_run: bool = False) -> None:
    """Step 5: ensure python-dotenv>=1.0.0 is in pyproject.toml."""
    pyproject = recipe_dir / "pyproject.toml"
    if ensure_python_dotenv_dependency(pyproject, dry_run=dry_run):
        verb = "Would add" if dry_run else "Added"
        print(
            f"[{_tag(dry_run)}] {verb} python-dotenv>=1.0.0 to pyproject.toml "
            "dependencies."
        )
    elif pyproject.exists():
        print("[PASS] pyproject.toml already includes python-dotenv — skipped.")
    else:
        print("[WARN] pyproject.toml not found — skipped.")


def run_step_model_names(
    recipe_dir: Path,
    py_files: list[Path],
    env_example: Path,
    dry_run: bool = False,
) -> None:
    """Step 6: detect hardcoded model strings, replace with os.getenv()."""
    model_hits = extract_hardcoded_models(py_files)
    if not model_hits:
        print("\n[PASS] No hardcoded model names detected.")
        return

    all_model_strings: set[str] = {
        model_str
        for file_hits in model_hits.values()
        for _lineno, model_str in file_hits
    }
    name_map = assign_model_var_names(all_model_strings)

    print("\n[INFO] Detected hardcoded model name(s):")
    for py_file, file_hits in model_hits.items():
        for lineno, model_str in file_hits:
            print(
                f"       {py_file.relative_to(recipe_dir)}:{lineno}"
                f' — "{model_str}" → {name_map[model_str]}'
            )

    substituted = replace_hardcoded_models(
        py_files, model_hits, name_map, dry_run=dry_run
    )
    if not substituted:
        return

    vars_to_add = {
        var_name: model_str for model_str, var_name in substituted.items()
    }
    added_models = update_env_example(env_example, vars_to_add, dry_run=dry_run)

    replace_verb = "Would replace" if dry_run else "Replaced"
    for model_str, var_name in substituted.items():
        print(
            f'[{_tag(dry_run)}] {replace_verb} hardcoded "{model_str}" with'
            f' os.getenv("{var_name}") in source.'
        )
    if added_models:
        add_verb = "Would add" if dry_run else "Added"
        for var in added_models:
            print(
                f"[{_tag(dry_run)}] {add_verb} {var}={vars_to_add[var]} "
                "to .env.example."
            )
    else:
        print("[PASS] All MODEL_NAME_* vars already in .env.example — skipped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a Python recipe and ensure all env vars are declared in "
            ".env.example, loaded via load_dotenv(), and python-dotenv is "
            "listed in pyproject.toml."
        )
    )
    parser.add_argument(
        "--recipe-dir",
        required=True,
        help="Path to the root of the Python recipe (e.g. contrib/my-recipe)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would change; do not modify any files.",
    )
    args = parser.parse_args()
    dry_run = args.dry_run

    recipe_dir = Path(args.recipe_dir).resolve()
    if not recipe_dir.is_dir():
        print(
            f"[ERROR] Recipe directory not found: {recipe_dir}", file=sys.stderr
        )
        sys.exit(1)

    print(f"\n{'=' * 50}")
    print("  extract-python-environment-variables")
    print(f"  Recipe: {recipe_dir}")
    if dry_run:
        print("  MODE: dry-run (no files will be modified)")
    print(f"{'=' * 50}\n")

    py_files = find_python_files(recipe_dir)
    print(f"[INFO] Scanning {len(py_files)} Python file(s) (tests/ excluded):")
    for f in py_files:
        print(f"       {f.relative_to(recipe_dir)}")

    env_example, _ = run_step_env_vars(recipe_dir, py_files, dry_run=dry_run)
    run_step_load_dotenv(recipe_dir, dry_run=dry_run)
    run_step_pyproject(recipe_dir, dry_run=dry_run)
    run_step_model_names(recipe_dir, py_files, env_example, dry_run=dry_run)

    print(f"\n{'=' * 50}")
    if dry_run:
        print("  Done (dry-run — no files were modified).")
    else:
        print("  Done.")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
