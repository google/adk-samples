#!/usr/bin/env python3
"""
Reject changes made inside retired recipe folders.

Recipes used to live under `<language>/agents/<recipe>` at the repo root
(e.g. `python/agents/academic-research`). Those roots are closed: new
recipes and changes to existing ones belong in
`contrib/<language>/<recipe>`. The retired roots are listed under
`frozen_paths` in .github/policy.yml.

Reads changed file paths from stdin (one per line) — the same interface as
tools/affected_recipes.py — and exits non-zero if any of them sit inside a
frozen path. Offending files are grouped by the recipe directory they
belong to, so a large PR produces one annotation per recipe rather than
one per file.

Callers should pass only ADDED and MODIFIED paths (`git diff
--diff-filter=AM`). Deletions and renames must be excluded so that
migrating a recipe out of a retired folder is not itself blocked.

Usage:
    git diff --diff-filter=AM --name-only origin/main...HEAD | \\
        uv run python tools/check_frozen_paths.py

Exit codes:
    0  no changes inside a retired folder
    1  at least one change inside a retired folder
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from ci_message import Diagnostic, Doc, guard, report

REPO_ROOT = Path(__file__).parent.parent
POLICY_PATH = REPO_ROOT / ".github" / "policy.yml"

# Where recipes live now. Only used to build the "move it here" hint.
ACTIVE_CONTRIB_ROOT = "contrib"

# A MAX_LISTED cap used to truncate the human-readable summary at 20
# entries. `report` in ci_message prints every diagnostic in full, so the
# cap is gone: a contributor with 21 affected recipes needs all 21, not
# 20 and a count.


def load_frozen_paths(policy_path: Path = POLICY_PATH) -> list[str]:
    """Return the `frozen_paths` list from policy.yml, normalised.

    A missing or empty section yields an empty list, which makes this
    check a no-op rather than a hard failure.
    """
    with open(policy_path, encoding="utf-8") as f:
        policy = yaml.safe_load(f) or {}
    raw = policy.get("frozen_paths") or []
    return [p for p in (str(x).strip().strip("/") for x in raw) if p]


def frozen_prefix_for(path: str, frozen_paths: list[str]) -> str | None:
    """Return the frozen prefix `path` sits under, or None.

    Matching is on whole path components, so a frozen prefix of
    `python/agents` matches `python/agents/foo/bar.py` but never
    `python/agents-archive/foo.py`.
    """
    parts = path.strip().strip("/").split("/")
    for prefix in frozen_paths:
        prefix_parts = prefix.split("/")
        if parts[: len(prefix_parts)] == prefix_parts:
            return prefix
    return None


def retired_recipe_dir(path: str, prefix: str) -> str:
    """The recipe directory `path` belongs to inside a retired folder.

    `python/agents/foo/app/agent.py` with prefix `python/agents` gives
    `python/agents/foo`. A path with nothing after the prefix degrades to
    the prefix itself.
    """
    parts = path.strip().strip("/").split("/")
    depth = len(prefix.split("/"))
    if len(parts) > depth:
        return "/".join(parts[: depth + 1])
    return prefix


def suggested_destination(recipe_dir: str) -> str:
    """Where a retired recipe should be moved to.

    `python/agents/foo` -> `contrib/python/foo`. Falls back to a generic
    hint when the path is too short to name a recipe.
    """
    parts = recipe_dir.split("/")
    language = parts[0]
    if len(parts) >= 3:
        return f"{ACTIVE_CONTRIB_ROOT}/{language}/{parts[-1]}"
    return f"{ACTIVE_CONTRIB_ROOT}/{language}/<recipe>"


def find_violations(
    changed_files: list[str], frozen_paths: list[str]
) -> dict[str, str]:
    """Map each offending recipe directory to one example file in it.

    Insertion order follows the first time each recipe directory is seen,
    so output is stable for a stable input ordering.
    """
    violations: dict[str, str] = {}
    for line in changed_files:
        path = line.strip()
        if not path:
            continue
        prefix = frozen_prefix_for(path, frozen_paths)
        if prefix is None:
            continue
        violations.setdefault(retired_recipe_dir(path, prefix), path)
    return violations


def _report(violations: dict[str, str]) -> int:
    """Report every violation through the shared diagnostic helper.

    Nothing is truncated: `report` prints one annotation per diagnostic.
    """
    diagnostics = [
        Diagnostic(
            check="frozen-path",
            what=(
                f"'{recipe_dir}' is in a retired folder, which no longer "
                f"accepts changes."
            ),
            why=(
                "Recipes used to live at <language>/agents/<recipe>. Those "
                "roots are closed — they are listed under frozen_paths in "
                ".github/policy.yml — and every active recipe now lives "
                "under contrib/. A change to the retired copy is invisible "
                "to anyone actually using the recipe."
            ),
            how=(
                f"Move the recipe, then make your change at the new path:\n"
                f"  git mv {recipe_dir} {suggested_destination(recipe_dir)}\n"
                f"The move itself is not blocked: this check ignores "
                f"deletions and renames precisely so a migration can "
                f"proceed. The moved recipe then has to meet the current "
                f"contribution requirements, which the retired copy "
                f"predates — see docs/recipe-checklist.md."
            ),
            doc=Doc.RETIRED_FOLDER,
            file=example,
        )
        for recipe_dir, example in violations.items()
    ]
    count = len(diagnostics)
    noun = "recipe" if count == 1 else "recipes"
    return report(
        diagnostics,
        header=f"{count} {noun} in a retired folder",
        passed_message="No changes inside retired recipe folders.",
        next_step=(
            "Move each recipe to the destination shown above, then re-run.\n"
            "Requirements for the moved recipe: docs/recipe-checklist.md"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    frozen_paths = load_frozen_paths()
    if not frozen_paths:
        print("[SKIP] No frozen_paths configured in .github/policy.yml.")
        return 0

    changed_files = sys.stdin.read().splitlines()
    violations = find_violations(changed_files, frozen_paths)

    if not violations:
        print("[PASS] No changes inside retired recipe folders.")
        return 0

    return _report(violations)


if __name__ == "__main__":
    sys.exit(guard("check_frozen_paths.py", main))
