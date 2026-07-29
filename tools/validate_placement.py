#!/usr/bin/env python3
"""
Validate that every recipe sits at the path its root requires.

Recipes are located by their manifest.yaml, and the manifest's depth below
the recipe root tells you whether the recipe is in the right place:

    skills/<vertical>/<solution>/manifest.yaml     valid
    skills/<solution>/manifest.yaml                too shallow — no vertical
    skills/<vertical>/<solution>/x/manifest.yaml   too deep

The vertical (retail/, hr/, finance/) is mandatory under skills/: it
surfaces ownership and lets a team reason about its whole surface at a
glance. A solution dropped directly under skills/ has no owning vertical,
so it is rejected.

Scope: only roots listed in validate_manifest.NAMESPACE_REQUIRED_ROOTS are
checked. core/ and contrib/ still permit a flat <root>/<recipe> layout and
are deliberately left alone here — widen CHECKED_ROOTS once that layout is
retired.

This is a whole-tree scan rather than a diff, so it also catches a
misplacement that arrives some other way (a rename, a bad merge). A missing
root is not an error: skills/ need not exist yet.

Usage:
    uv run python tools/validate_placement.py

Exit codes:
    0 — every recipe is correctly placed
    1 — one or more recipes are misplaced
"""

from __future__ import annotations

import sys
from pathlib import Path

import validate_manifest as vm

REPO_ROOT = Path(__file__).parent.parent

# Roots this checker enforces. Kept deliberately narrow: these are the roots
# where a namespace is mandatory, so the expected depth is unambiguous.
CHECKED_ROOTS = sorted(vm.NAMESPACE_REQUIRED_ROOTS)

# Directory names never descended into while looking for manifests.
PRUNED_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".ruff_cache"}

# <root>/<namespace>/<recipe>/manifest.yaml
EXPECTED_PARTS = 4


def find_manifests(root: Path) -> list[Path]:
    """Every manifest.yaml beneath `root`, skipping vendored/build dirs."""
    if not root.is_dir():
        return []
    found: list[Path] = []
    for path in sorted(root.rglob(vm.MANIFEST_FILENAME)):
        if any(part in PRUNED_DIRS for part in path.parts):
            continue
        found.append(path)
    return found


def describe_violation(rel_parts: list[str]) -> str | None:
    """Return an error message for a manifest path, or None if it is valid.

    `rel_parts` are the manifest's path components relative to the repo
    root, e.g. ["skills", "retail", "store-ops", "manifest.yaml"].
    """
    if len(rel_parts) == EXPECTED_PARTS:
        return None

    root = rel_parts[0]
    recipe_dir = "/".join(rel_parts[:-1])
    expected = f"{root}/<vertical>/<solution>/{vm.MANIFEST_FILENAME}"

    if len(rel_parts) < EXPECTED_PARTS:
        solution = rel_parts[-2] if len(rel_parts) > 1 else "<solution>"
        return (
            f"'{recipe_dir}' sits directly under '{root}/' with no vertical. "
            f"Every {root}/ recipe must live at {expected} — move it to "
            f"'{root}/<vertical>/{solution}' (e.g. "
            f"'{root}/retail/{solution}'), choosing the vertical that owns "
            f"it."
        )

    return (
        f"'{recipe_dir}' is nested too deeply. Every {root}/ recipe must "
        f"live at {expected}, one directory below its vertical."
    )


def check_root(
    root_name: str,
    repo_root: Path = REPO_ROOT,
    scope: str | None = None,
) -> list[str]:
    """Return an error string for every misplaced recipe under a root.

    `scope`, when given, restricts reporting to manifests beneath that
    repo-relative path, so `validate placement core/python/foo` cannot fail
    on an unrelated problem elsewhere in the tree.
    """
    errors: list[str] = []
    for manifest in find_manifests(repo_root / root_name):
        rel = manifest.relative_to(repo_root)
        if scope and not (
            str(rel) == scope or str(rel).startswith(f"{scope}/")
        ):
            continue
        message = describe_violation(list(rel.parts))
        if message is None:
            continue
        print(f"::error file={rel}::{message}")
        errors.append(message)
    return errors


def main(scope: str | None = None) -> int:
    """Check every root in CHECKED_ROOTS, or just the part under `scope`.

    Placement is a whole-tree property, so the useful invocation is the
    unscoped one that CI runs. A scope is honoured anyway — it is what the
    other `validate` subcommands accept, and narrowing keeps
    `uv run validate <a-core-recipe>` from failing on an unrelated skill.
    """
    prefix = scope.strip("/") if scope else None
    all_errors: list[str] = []
    for root_name in CHECKED_ROOTS:
        # A scope pointing outside this root (e.g. core/python/foo) means
        # there is nothing here for it to say.
        if prefix and not (
            prefix == root_name or prefix.startswith(f"{root_name}/")
        ):
            continue
        root_path = REPO_ROOT / root_name
        if not root_path.is_dir():
            print(f"[SKIP] '{root_name}/' does not exist.")
            continue
        # Pass REPO_ROOT explicitly: check_root's default argument is bound
        # at import time, so relying on it would ignore any later rebinding
        # of the module global (which is how the tests point at a fixture).
        errors = check_root(root_name, repo_root=REPO_ROOT, scope=prefix)
        if errors:
            print(f"\n[FAIL] Misplaced recipes under '{root_name}/':\n")
            for message in errors:
                print(f"  - {message}")
            all_errors.extend(errors)
        else:
            print(f"[PASS] All '{root_name}/' recipes are correctly placed.")

    if not all_errors:
        return 0

    count = len(all_errors)
    noun = "recipe" if count == 1 else "recipes"
    print("")
    print("=" * 60)
    print(f"  ACTION REQUIRED: {count} misplaced {noun}")
    print("=" * 60)
    print("")
    print("Move each recipe to the path shown above, then re-run:")
    print("")
    print("  uv run python tools/validate_placement.py")
    print("")
    return 1


if __name__ == "__main__":
    sys.exit(main())
