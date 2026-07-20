"""
Checks that every environment variable read by a recipe's Python source is
declared in the recipe's .env.example.

AST-based: understands os.getenv(), os.environ.get(), os.environ[...],
and their import-alias forms (from os import getenv; from os.environ import
get) regardless of how the calls are formatted or split across lines.
An allowlist of well-known OS/CI variables suppresses false positives for
variables that legitimately do not belong in .env.example (HOME, PATH, CI,
GITHUB_*, etc.).

Usage: python3 check_env_vars.py <recipe-dir>

Output format (one record per line, for the shell caller to parse):
  PASS::<path>::<message>
  FAIL::<path>::<message>

Exits 0 always.  The workflow decides pass/fail from the emitted records so
that a missing .env.example (caught by a separate required-files check)
does not produce a redundant error here.

MAINTENANCE NOTE — keep in sync with the extract-python-environment-variables
skill (.agents/skills/extract-python-environment-variables/scripts/
extract_env_vars.py).  This script READS/validates; that one REWRITES.  They
are intentionally separate tools but must stay semantically aligned: if you
add a detection pattern or modify the allowlist here, mirror the change there.
"""

import ast
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowlist
#
# Variables in this set (or matching a prefix below) are provided by the OS,
# the CI runtime, or the execution environment.  They should NOT appear in
# .env.example because their values vary per machine and are never
# recipe-specific secrets or configuration.  Adding a name here suppresses
# the FAIL that would otherwise fire when a recipe reads it but does not
# declare it.
#
# Keep this list conservative.  When in doubt, do NOT allowlist — let the
# recipe declare the variable and explain why it exists.
# ---------------------------------------------------------------------------
_ALLOWLIST: frozenset[str] = frozenset(
    {
        # POSIX core
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "PWD",
        "OLDPWD",
        "PATH",
        "TMPDIR",
        "TEMP",
        "TMP",
        # Locale / terminal
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TERM",
        "TERM_PROGRAM",
        "COLORTERM",
        "TZ",
        "EDITOR",
        "VISUAL",
        "PAGER",
        # Python runtime flags
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "PYTHONSTARTUP",
        # Virtual-environment managers
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        # Common CI / test flags
        "CI",
        "CONTINUOUS_INTEGRATION",
        "DEBUG",
        "PORT",
        "HOST",
        "HOSTNAME",
        # ADK runnability-test sentinel — set by the test harness, not the
        # developer, so it should not appear in .env.example.
        "INTEGRATION_TEST",
    }
)

# Variable-name prefixes that are automatically allowed without being listed
# individually above.
_ALLOWLIST_PREFIXES: tuple[str, ...] = (
    "GITHUB_",  # GitHub Actions context variables (GITHUB_TOKEN, etc.)
    "RUNNER_",  # GitHub Actions runner variables
    "ACTIONS_",  # GitHub Actions built-ins
)


def _is_allowed(name: str) -> bool:
    return name in _ALLOWLIST or any(
        name.startswith(p) for p in _ALLOWLIST_PREFIXES
    )


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------


class _EnvVarCollector(ast.NodeVisitor):
    """Collects env-variable names from canonical os/os.environ read patterns.

    Detected patterns
    -----------------
    os.getenv("VAR")                  Attribute call on 'os'
    os.environ.get("VAR")             Chained attribute call
    os.environ["VAR"]                 Subscript on os.environ
    getenv("VAR")                     After: from os import getenv [as ...]
    get("VAR")                        After: from os.environ import get [as ...]

    Intentionally NOT detected (to avoid false positives)
    -------------------------------------------------------
    os.environ.setdefault("VAR", ...) — write/default, not a pure read
    os.environ.pop("VAR")             — deletion
    Dynamic keys: os.getenv(some_var) — cannot resolve statically
    Keyword-only calls: os.getenv(key="VAR") — extremely uncommon
    """

    def __init__(self) -> None:
        self.env_vars: set[str] = set()
        # Aliases introduced by `from os import getenv [as foo]`
        self._getenv_aliases: set[str] = set()
        # Aliases introduced by `from os.environ import get [as foo]`
        self._environ_get_aliases: set[str] = set()

    # ------------------------------------------------------------------
    # Import tracking
    # ------------------------------------------------------------------

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "os":
            for alias in node.names:
                if alias.name == "getenv":
                    self._getenv_aliases.add(alias.asname or alias.name)
        elif node.module == "os.environ":
            for alias in node.names:
                if alias.name == "get":
                    self._environ_get_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Call patterns
    # ------------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        # os.getenv("VAR")
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "getenv"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
        ):
            self._capture_first_str(node)

        # os.environ.get("VAR")
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "os"
        ):
            self._capture_first_str(node)

        # getenv("VAR")  after `from os import getenv`
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in self._getenv_aliases
        ):
            self._capture_first_str(node)

        # get("VAR")  after `from os.environ import get`
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in self._environ_get_aliases
        ):
            self._capture_first_str(node)

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Subscript pattern
    # ------------------------------------------------------------------

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # os.environ["VAR"]
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "environ"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
        ):
            # Python 3.9+: node.slice is the value directly.
            # Python 3.8:  node.slice is ast.Index(value=...).
            slice_node = node.slice
            if isinstance(slice_node, ast.Index):  # type: ignore[attr-defined]
                slice_node = slice_node.value  # type: ignore[attr-defined]
            if isinstance(slice_node, ast.Constant) and isinstance(
                slice_node.value, str
            ):
                self.env_vars.add(slice_node.value)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _capture_first_str(self, node: ast.Call) -> None:
        """Add the first positional string-literal argument to env_vars."""
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self.env_vars.add(node.args[0].value)


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------


_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        # Generated / tool directories — contain third-party code, not recipe
        # source, and scanning them produces enormous false-positive floods.
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        # Test directory — checked separately; runnability tests intentionally
        # use stub env vars that shouldn't be declared in .env.example.
        "tests",
    }
)


def _collect_used_vars(recipe_dir: Path) -> set[str]:
    """AST-parse all non-test, non-venv Python files; return env-var names read."""
    used: set[str] = set()
    for py_file in sorted(recipe_dir.rglob("*.py")):
        # Exclude anything under a directory that appears in the exclusion set.
        if any(part in _EXCLUDED_DIRS for part in py_file.parts):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            # Unparseable files are skipped silently; a separate lint step
            # (ruff) catches syntax errors in the same CI run.
            continue
        collector = _EnvVarCollector()
        collector.visit(tree)
        used |= collector.env_vars
    return used


def _parse_env_example(env_example: Path) -> set[str]:
    """Return the set of variable names declared in .env.example."""
    defined: set[str] = set()
    for raw in env_example.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=", line)
        if m:
            defined.add(m.group(1))
    return defined


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def emit(kind: str, file: Path, msg: str) -> None:
    """Print one PASS/FAIL record.  Newlines collapsed for line-by-line parsing."""
    print(f"{kind}::{file}::{msg.replace(chr(10), ' ')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <recipe-dir>", file=sys.stderr)
        return 2

    recipe_dir = Path(sys.argv[1])
    env_example = recipe_dir / ".env.example"

    if not env_example.is_file():
        # A separate required-files check already reports this; stay silent
        # here so we don't double-report the same missing file.
        return 0

    defined_vars = _parse_env_example(env_example)
    used_vars = _collect_used_vars(recipe_dir)

    missing = sorted(
        v for v in used_vars if v not in defined_vars and not _is_allowed(v)
    )

    if missing:
        for var in missing:
            emit(
                "FAIL",
                env_example,
                f"Environment variable '{var}' is read by Python source but"
                f" not declared in .env.example. Add it (with a TODO"
                f" placeholder as the value) and load it with load_dotenv().",
            )
        return 0

    if not used_vars:
        emit(
            "PASS",
            env_example,
            "No environment variable reads detected in Python source.",
        )
    else:
        n_checked = len(used_vars)
        n_allowed = sum(1 for v in used_vars if _is_allowed(v))
        emit(
            "PASS",
            env_example,
            f"All environment variables read in Python source are declared in"
            f" .env.example ({n_checked} detected,"
            f" {n_allowed} in OS allowlist, {n_checked - n_allowed} declared).",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
