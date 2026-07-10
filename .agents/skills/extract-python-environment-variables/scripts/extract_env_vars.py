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


def extract_env_vars(py_files: list[Path]) -> dict[str, str | None]:
    """
    Walk each file's AST and collect env var names + optional inline defaults.

    Detects:
      os.environ["VAR"]
      os.environ.get("VAR")
      os.environ.get("VAR", "default")
      os.getenv("VAR")
      os.getenv("VAR", "default")

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
                f"[WARN] Could not parse {py_file} — skipping.",
                file=sys.stderr,
            )
            continue

        for node in ast.walk(tree):
            var_name: str | None = None
            default: str | None = None

            # ------------------------------------------------------------------
            # os.getenv("VAR") / os.getenv("VAR", "default")
            # ------------------------------------------------------------------
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "getenv"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                var_name = node.args[0].value
                if (
                    len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    default = node.args[1].value

            # ------------------------------------------------------------------
            # os.environ.get("VAR") / os.environ.get("VAR", "default")
            # ------------------------------------------------------------------
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "os"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                var_name = node.args[0].value
                if (
                    len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    default = node.args[1].value

            # ------------------------------------------------------------------
            # os.environ["VAR"]  (subscript)
            # ------------------------------------------------------------------
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "os"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                var_name = node.slice.value

            # Only keep SCREAMING_SNAKE_CASE names (conventional env var style)
            if var_name and re.match(r"^[A-Z_][A-Z0-9_]*$", var_name):
                # Prefer non-None default if we've seen the var before
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

    block = "\n# Environment variables extracted by extract-python-environment-variables\n"
    for var in sorted(to_add):
        value = to_add[var] if to_add[var] is not None else PLACEHOLDER
        block += f"{var}={value}\n"

    env_example.write_text(current + block, encoding="utf-8")
    return sorted(to_add.keys())


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
    inject_block = (
        f"\n{LOAD_DOTENV_IMPORT}\n"
        f"\n{LOAD_DOTENV_SNIPPET}\n"
    )

    # Find the index of the last absolute import line so we can insert after it.
    # Relative imports (from .something) must come AFTER load_dotenv() so that
    # the env is populated before any package module-level code runs.
    last_import_idx: int = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_absolute_import = (
            stripped.startswith("import ")
            or (stripped.startswith("from ") and not stripped.startswith("from ."))
        )
        if is_absolute_import:
            last_import_idx = i

    if last_import_idx >= 0:
        lines.insert(last_import_idx + 1, inject_block)
    else:
        # No imports found — insert after the license/comment header block
        license_end = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("#") or not line.strip():
                license_end = i + 1
            else:
                break
        lines.insert(license_end, inject_block)

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
        # Insert new dep before the closing bracket, preserving indentation
        inner = m.group(2)
        # Detect the indentation used by existing entries
        indent_match = re.search(r"\n(\s+)", inner)
        indent = indent_match.group(1) if indent_match else "    "
        return (
            m.group(1)
            + inner
            + f'{indent}"python-dotenv>=1.0.0",\n'
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
            f"[ERROR] Recipe directory not found: {recipe_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  extract-python-environment-variables")
    print(f"  Recipe: {recipe_dir}")
    print(f"{'='*50}\n")

    # ------------------------------------------------------------------
    # Step 1 — Find Python files
    # ------------------------------------------------------------------
    py_files = find_python_files(recipe_dir)
    print(f"[INFO] Scanning {len(py_files)} Python file(s) (tests/ excluded):")
    for f in py_files:
        print(f"       {f.relative_to(recipe_dir)}")

    # ------------------------------------------------------------------
    # Step 2 — Extract env vars
    # ------------------------------------------------------------------
    env_vars = extract_env_vars(py_files)
    if env_vars:
        print(f"\n[INFO] Detected {len(env_vars)} environment variable(s):")
        for var in sorted(env_vars):
            default = env_vars[var]
            suffix = f"  (default: {default!r})" if default is not None else ""
            print(f"       {var}{suffix}")
    else:
        print("\n[INFO] No environment variable reads detected.")

    # ------------------------------------------------------------------
    # Step 3 — Update .env.example
    # ------------------------------------------------------------------
    env_example = recipe_dir / ".env.example"
    added = update_env_example(env_example, env_vars)
    if added:
        print(
            f"\n[PASS] Added {len(added)} variable(s) to .env.example: "
            + ", ".join(added)
        )
    else:
        print("\n[PASS] .env.example is already up to date — no variables added.")

    # ------------------------------------------------------------------
    # Step 4 — Inject load_dotenv() into package __init__.py
    # ------------------------------------------------------------------
    init_py = find_package_init(recipe_dir)
    if init_py:
        modified = inject_load_dotenv(init_py)
        rel = init_py.relative_to(recipe_dir)
        if modified:
            print(f"[PASS] Injected load_dotenv() bootstrap into {rel}")
        else:
            print(f"[PASS] load_dotenv() already present in {rel} — skipped.")
    else:
        print(
            "[WARN] No Python package (subdirectory with __init__.py) found. "
            "load_dotenv() injection skipped."
        )

    # ------------------------------------------------------------------
    # Step 5 — Update pyproject.toml
    # ------------------------------------------------------------------
    pyproject = recipe_dir / "pyproject.toml"
    modified = ensure_python_dotenv_dependency(pyproject)
    if modified:
        print("[PASS] Added python-dotenv>=1.0.0 to pyproject.toml dependencies.")
    elif pyproject.exists():
        print("[PASS] pyproject.toml already includes python-dotenv — skipped.")
    else:
        print("[WARN] pyproject.toml not found — skipped.")

    print(f"\n{'='*50}")
    print(f"  Done.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
