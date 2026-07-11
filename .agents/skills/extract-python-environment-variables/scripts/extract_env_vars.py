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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLACEHOLDER = "<TODO: update-this-value>"

# Model name prefixes that indicate a hardcoded model identifier.
# Kept in sync with validate-python-sample.yml.
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
# Load the .env file
if not load_dotenv():
    raise FileNotFoundError(
        "Critical Error: No .env file found. "
        "Make sure to copy .env.example to .env and update the values."
    )"""


# ---------------------------------------------------------------------------
# Step 1: Find Python files (excluding tests/)
# ---------------------------------------------------------------------------


def find_python_files(recipe_dir: Path) -> list[Path]:
    """Return all .py files under recipe_dir, skipping tests/ directories."""
    return [
        p
        for p in sorted(recipe_dir.rglob("*.py"))
        if "tests" not in p.relative_to(recipe_dir).parts
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
    env_example: Path, env_vars: dict[str, str | None]
) -> list[str]:
    """
    Append variables not yet in .env.example.
    Returns the list of variable names that were added.
    """
    existing = read_defined_vars(env_example)
    to_add = {k: v for k, v in env_vars.items() if k not in existing}

    if not to_add:
        return []

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


# ---------------------------------------------------------------------------
# Step 4: Inject load_dotenv() into package __init__.py
# ---------------------------------------------------------------------------


def find_package_init(recipe_dir: Path) -> Path | None:
    """
    Return the __init__.py of the top-level Python package inside recipe_dir.
    Looks for immediate subdirectories that contain __init__.py.
    """
    for candidate in sorted(recipe_dir.iterdir()):
        if candidate.is_dir() and (candidate / "__init__.py").exists():
            return candidate / "__init__.py"
    return None


def inject_load_dotenv(init_py: Path) -> bool:
    """
    Ensure load_dotenv import + bootstrap snippet exist in __init__.py.
    Returns True if the file was modified.
    """
    content = init_py.read_text(encoding="utf-8")

    if "load_dotenv" in content:
        return False  # Already present — nothing to do

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

    init_py.write_text("".join(lines), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Step 5: Ensure python-dotenv in pyproject.toml
# ---------------------------------------------------------------------------


def ensure_python_dotenv_dependency(pyproject: Path) -> bool:
    """
    Add python-dotenv>=1.0.0 to [project] dependencies if absent.
    Returns True if the file was modified.
    """
    if not pyproject.exists():
        return False

    content = pyproject.read_text(encoding="utf-8")
    if "python-dotenv" in content:
        return False

    # Regex: match the dependencies = [ ... ] block (multiline, non-greedy)
    def inserter(m: re.Match) -> str:
        inner = m.group(2)
        indent_match = re.search(r"\n(\s+)", inner)
        indent = indent_match.group(1) if indent_match else "    "
        inner_stripped = inner.rstrip()
        # Ensure the last existing entry ends with a comma (handles single-line
        # arrays like  dependencies = ["pkg"]  where no trailing comma exists).
        if inner_stripped and not inner_stripped.endswith(","):
            inner_stripped += ","
        is_multiline = "\n" in inner
        sep = f"\n{indent}" if is_multiline else " "
        tail = "\n" if is_multiline else ""
        return (
            m.group(1)
            + inner_stripped
            + f'{sep}"python-dotenv>=1.0.0",'
            + tail
            + m.group(3)
        )

    new_content = re.sub(
        r"(dependencies\s*=\s*\[)(.*?)(\])",
        inserter,
        content,
        count=1,
        flags=re.DOTALL,
    )

    if new_content == content:
        return False

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
    any known model prefix (same list as validate-python-sample.yml).
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


def assign_model_var_names(model_strings: set[str]) -> dict[str, str]:
    """
    Assign a standardised MODEL_NAME_* env var name to each unique model string.

    Rules (applied to the sorted list for determinism):
      - A model string containing "embed" → MODEL_NAME_EMBEDDING
      - The first non-embedding model     → MODEL_NAME
      - Additional non-embedding models   → MODEL_NAME_2, MODEL_NAME_3, …

    Returns:
      {model_string: env_var_name}
    """
    mapping: dict[str, str] = {}
    non_embedding_idx = 0

    for model_str in sorted(model_strings):
        if "embed" in model_str.lower():
            mapping[model_str] = "MODEL_NAME_EMBEDDING"
        else:
            if non_embedding_idx == 0:
                mapping[model_str] = "MODEL_NAME"
            else:
                mapping[model_str] = f"MODEL_NAME_{non_embedding_idx + 1}"
            non_embedding_idx += 1

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

        # Apply in reverse order so earlier offsets stay valid
        replacements.sort(key=lambda x: x[0], reverse=True)
        chars = list(source)
        for start, end, new_text in replacements:
            chars[start:end] = list(new_text)
        modified = "".join(chars)

        # Ensure `import os` is present, placed after license + docstring.
        # Include a trailing blank line so the import block is well-formatted.
        if "import os" not in modified:
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


def run_step_env_vars(
    recipe_dir: Path, py_files: list[Path]
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
    added = update_env_example(env_example, env_vars)
    if added:
        print(
            f"\n[PASS] Added {len(added)} variable(s) to .env.example: "
            + ", ".join(added)
        )
    else:
        print(
            "\n[PASS] .env.example is already up to date — no variables added."
        )

    return env_example, env_vars


def run_step_load_dotenv(recipe_dir: Path) -> None:
    """Step 4: inject load_dotenv() bootstrap into the package __init__.py."""
    init_py = find_package_init(recipe_dir)
    if not init_py:
        print(
            "[WARN] No Python package (subdirectory with __init__.py) found. "
            "load_dotenv() injection skipped."
        )
        return
    rel = init_py.relative_to(recipe_dir)
    if inject_load_dotenv(init_py):
        print(f"[PASS] Injected load_dotenv() bootstrap into {rel}")
    else:
        print(f"[PASS] load_dotenv() already present in {rel} — skipped.")


def run_step_pyproject(recipe_dir: Path) -> None:
    """Step 5: ensure python-dotenv>=1.0.0 is in pyproject.toml."""
    pyproject = recipe_dir / "pyproject.toml"
    if ensure_python_dotenv_dependency(pyproject):
        print(
            "[PASS] Added python-dotenv>=1.0.0 to pyproject.toml dependencies."
        )
    elif pyproject.exists():
        print("[PASS] pyproject.toml already includes python-dotenv — skipped.")
    else:
        print("[WARN] pyproject.toml not found — skipped.")


def run_step_model_names(
    recipe_dir: Path, py_files: list[Path], env_example: Path
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

    substituted = replace_hardcoded_models(py_files, model_hits, name_map)
    if not substituted:
        return

    vars_to_add = {
        var_name: model_str for model_str, var_name in substituted.items()
    }
    added_models = update_env_example(env_example, vars_to_add)

    for model_str, var_name in substituted.items():
        print(
            f'[PASS] Replaced hardcoded "{model_str}" with'
            f' os.getenv("{var_name}") in source.'
        )
    if added_models:
        for var in added_models:
            print(f"[PASS] Added {var}={vars_to_add[var]} to .env.example.")
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
    args = parser.parse_args()

    recipe_dir = Path(args.recipe_dir).resolve()
    if not recipe_dir.is_dir():
        print(
            f"[ERROR] Recipe directory not found: {recipe_dir}", file=sys.stderr
        )
        sys.exit(1)

    print(f"\n{'=' * 50}")
    print("  extract-python-environment-variables")
    print(f"  Recipe: {recipe_dir}")
    print(f"{'=' * 50}\n")

    py_files = find_python_files(recipe_dir)
    print(f"[INFO] Scanning {len(py_files)} Python file(s) (tests/ excluded):")
    for f in py_files:
        print(f"       {f.relative_to(recipe_dir)}")

    env_example, _ = run_step_env_vars(recipe_dir, py_files)
    run_step_load_dotenv(recipe_dir)
    run_step_pyproject(recipe_dir)
    run_step_model_names(recipe_dir, py_files, env_example)

    print(f"\n{'=' * 50}")
    print("  Done.")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
