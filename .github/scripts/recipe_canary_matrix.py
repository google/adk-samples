#!/usr/bin/env python3
"""
Build the job matrix for .github/workflows/recipe-canary.yml.

Emits one entry per (recipe, python-version) pair the canary should exercise,
as a JSON array on stdout:

    [{"recipe": "core/python/deep-search", "python": "3.11"}, ...]

Why two Python versions per recipe
----------------------------------
`python-tests.yml` pins Python 3.11 and nothing else, so a recipe declaring
`requires-python = ">=3.11,<3.14"` has its claim to support 3.12 and 3.13
tested nowhere. That is not theoretical: `core/python/rag-agent-search`
advertises `<3.14` and cannot be imported on 3.13 at all — the google-genai
2.10.0 it pins ships a module with a misplaced `from __future__` import, a
hard SyntaxError. It passes cleanly on 3.11.

So the canary tests the floor AND the ceiling of what each recipe claims:
the version everyone actually uses, plus the highest one it promises to work
on. A recipe that only claims 3.11 gets a single job.

This is the rot the canary exists for. A frozen lockfile does not decay on
its own — the interpreter moves underneath it.

Scope
-----
Python recipes under core/, contrib/ and skills/, discovered by manifest.yaml
declaring `language: python`. Recipes marked `status: inactive` are INCLUDED:
a recipe on the retirement path still needs to be noticed if it starts
passing again, and skipping it would make "fixed but never reactivated"
invisible until deletion.

Usage:
    python recipe_canary_matrix.py            # every recipe
    python recipe_canary_matrix.py --recipe core/python/deep-search
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SCAN_ROOTS = ["core", "contrib", "skills"]

SKIP_DIRS = {
    ".venv",
    "node_modules",
    ".git",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    "dist",
    "build",
}

# The floor every recipe must accept (AGENTS.md: "Minimum python version:
# 3.11", and CI pins it).
FLOOR = "3.11"

# The highest minor the canary is willing to provision. A recipe with no
# upper bound declares support for everything, which cannot be tested, so the
# claim is checked up to here and no further. Raise it when the runners and
# the ecosystem have caught up with a new release — and expect new failures,
# because that is the point.
MAX_TESTABLE_MINOR = 13

# Recipes the canary deliberately does not run.
#
# core/rag-agent-search and core/rag-vector-search are legacy flat-path
# duplicates of their core/python/* counterparts — same recipe, maintained in
# parallel, scheduled for deletion. Running both copies files two issues for
# one problem and @-mentions the owner twice.
#
# REMOVE THESE ENTRIES when the duplicates are deleted. If the paths are gone
# and these lines remain they are merely dead, but a stale skip that silently
# matched a real recipe would not be, so the accompanying test asserts every
# entry still exists.
SKIP_RECIPES = {
    "core/rag-agent-search",
    "core/rag-vector-search",
}


def _is_python_recipe(manifest_path: Path) -> bool:
    """`language: python` in manifest.yaml, without requiring a YAML parser.

    Mirrors the tolerant matcher in python-tests.yml: optional quotes, any
    case, an optional trailing comment. Kept regex-based so the canary matrix
    can be produced with the standard library alone.
    """
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(
        re.search(
            r"""^[ \t]*language:[ \t]*["']?python["']?[ \t]*(?:\#.*)?$""",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
    )


def discover_recipes(repo_root: Path | None = None) -> list[str]:
    """Repo-relative paths of every Python recipe root, sorted."""
    root = REPO_ROOT if repo_root is None else repo_root
    found: list[str] = []
    for root_name in SCAN_ROOTS:
        root_path = root / root_name
        if not root_path.is_dir():
            continue
        for manifest in sorted(root_path.rglob("manifest.yaml")):
            if any(part in SKIP_DIRS for part in manifest.parts):
                continue
            if not _is_python_recipe(manifest):
                continue
            rel = manifest.parent.relative_to(root).as_posix()
            if rel in SKIP_RECIPES:
                continue
            found.append(rel)
    return sorted(found)


def _upper_bound_minor(requires_python: str) -> int | None:
    """Highest 3.x minor admitted by an exclusive upper bound, if any.

    Only `<` and `<=` are interpreted, because they are the only forms the
    repo's recipes use and the only ones with an unambiguous "highest minor I
    claim" reading. Anything else yields None and falls back to
    MAX_TESTABLE_MINOR — the conservative direction is to test MORE, since a
    passing extra job costs a few minutes and a missed one costs a broken
    recipe nobody hears about.
    """
    best: int | None = None
    for op, ver in re.findall(r"(<=|<)\s*3\.(\d+)(?:\.\d+)?", requires_python):
        minor = int(ver)
        # `<3.14` admits 3.13; `<=3.13` admits 3.13.
        highest = minor - 1 if op == "<" else minor
        best = highest if best is None else min(best, highest)
    return best


def python_targets(recipe_dir: Path) -> list[str]:
    """The Python versions the canary should run for one recipe.

    Always the floor. Plus the ceiling the recipe claims, when that is higher
    and we can actually provision it.
    """
    targets = [FLOOR]
    pyproject = recipe_dir / "pyproject.toml"
    if not pyproject.is_file():
        return targets

    try:
        with open(pyproject, "rb") as handle:
            data = tomllib.load(handle)
    except (tomllib.TOMLDecodeError, OSError):
        # A pyproject that will not parse is reported by the validate
        # workflow, which owns that complaint. Fall back to the floor.
        return targets

    requires = (data.get("project") or {}).get("requires-python")
    if not isinstance(requires, str):
        return targets

    declared = _upper_bound_minor(requires)
    ceiling = MAX_TESTABLE_MINOR if declared is None else declared
    ceiling = min(ceiling, MAX_TESTABLE_MINOR)

    floor_minor = int(FLOOR.split(".")[1])
    if ceiling > floor_minor:
        targets.append(f"3.{ceiling}")
    return targets


def build_matrix(
    repo_root: Path | None = None, only: str | None = None
) -> list[dict[str, str]]:
    root = REPO_ROOT if repo_root is None else repo_root
    recipes = [only] if only else discover_recipes(root)
    matrix: list[dict[str, str]] = []
    for recipe in recipes:
        for version in python_targets(root / recipe):
            matrix.append({"recipe": recipe, "python": version})
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        help="Limit the matrix to one repo-relative recipe path.",
    )
    args = parser.parse_args(argv)

    matrix = build_matrix(only=args.recipe)
    if not matrix:
        print(
            "No Python recipes found. The canary refuses to report success "
            "on an empty scan — that is indistinguishable from every recipe "
            "having vanished.",
            file=sys.stderr,
        )
        return 1

    json.dump(matrix, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
