#!/usr/bin/env python3
"""
Validates manifest.yaml files in recipe directories under core/ and contrib/.

Checks:
  - Every recipe directory has a manifest.yaml file.
  - Every manifest.yaml is valid YAML.
  - Every manifest.yaml conforms to the schema at
    .github/schemas/manifest-schema.json.
  - ownership.team and ownership.poc are not left as placeholder values.

Usage:
  # Validate all recipes (both core/ and contrib/):
  python3 tools/validate_manifest.py
  python3 tools/validate_manifest.py all

  # Validate only core/ or contrib/:
  python3 tools/validate_manifest.py core
  python3 tools/validate_manifest.py contrib

  # Validate a single recipe:
  python3 tools/validate_manifest.py core/rag-agent-search

  Dependencies are managed in pyproject.toml. Run `uv sync` once before using.

Exit codes:
  0 — all manifests present and valid
  1 — one or more manifests missing or invalid
"""

import argparse
import json
import sys
from pathlib import Path

import jsonschema
import yaml

REPO_ROOT = Path(__file__).parent.parent
SCHEMA_PATH = REPO_ROOT / ".github" / "schemas" / "manifest-schema.json"
MANIFEST_FILENAME = "manifest.yaml"
RECIPE_ROOTS = ["core", "contrib"]

OWNERSHIP_TEAM_PLACEHOLDER = "TODO: Replace with your team name"
OWNERSHIP_POC_PLACEHOLDER = "TODO: Replace with your GitHub user ID"

LANGUAGE_NAMESPACE_DIRS = {"python", "java", "go", "typescript", "kotlin"}


def is_recipe_dir(path: Path) -> bool:
    """A recipe directory is any immediate subdirectory that contains
    more than just a README.md, and is not a language namespace directory."""
    if not path.is_dir() or path.name.startswith("."):
        return False
    # Language namespace dirs (e.g. core/python/) are not recipes themselves;
    # they are containers whose children are the actual recipes.
    if path.name in LANGUAGE_NAMESPACE_DIRS:
        return False
    children = [
        p
        for p in path.iterdir()
        if not p.name.startswith(".") and p.name != "README.md"
    ]
    if not children:
        return False
    # A directory whose non-hidden children are exclusively language namespace
    # dirs is itself a container (e.g. core/harnesses/), not a recipe.
    if all(p.is_dir() and p.name in LANGUAGE_NAMESPACE_DIRS for p in children):
        return False
    return True


def load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(manifest_path: Path, schema: dict) -> list[str]:
    """Returns a list of error strings. Empty list means valid."""
    errors = []
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        errors.append(f"YAML parse error: {e}")
        return errors

    if data is None:
        errors.append("manifest.yaml is empty")
        return errors

    validator = jsonschema.Draft7Validator(schema)
    for err in sorted(validator.iter_errors(data), key=str):
        errors.append(f"  [{err.json_path}] {err.message}")

    # Check that placeholder values have been replaced with real ones
    if isinstance(data, dict):
        ownership = data.get("ownership")
        if isinstance(ownership, dict):
            if ownership.get("team") == OWNERSHIP_TEAM_PLACEHOLDER:
                errors.append(
                    "  [ownership.team] is still set to the placeholder value "
                    f'"{OWNERSHIP_TEAM_PLACEHOLDER}". '
                    "Please replace it with a real team name."
                )
            if ownership.get("poc") == OWNERSHIP_POC_PLACEHOLDER:
                errors.append(
                    "  [ownership.poc] is still set to the placeholder value "
                    f'"{OWNERSHIP_POC_PLACEHOLDER}". '
                    "Please replace it with a real GitHub ID."
                )

        # A description left as a "TODO ..." placeholder (e.g. the scaffold
        # template's default) is long enough to satisfy the schema's
        # minLength, so it would otherwise slip through. Guard it explicitly,
        # mirroring the ownership checks above. A prefix match (rather than an
        # exact string) keeps this robust to wording changes and catches any
        # hand-written "TODO ..." description too.
        description = data.get("description")
        if isinstance(
            description, str
        ) and description.strip().upper().startswith("TODO"):
            errors.append(
                "  [description] is still a TODO placeholder. Please replace "
                "it with a real description of what the recipe demonstrates."
            )

    return errors


def _collect_scoped_path(scope: str) -> list[Path]:
    """Resolve a scope that points at a specific path (not a bare root).

    Handles a language namespace dir (recurse one level) or a single recipe
    directory. Exits the process on an invalid path.
    """
    target = REPO_ROOT / scope
    if not target.exists():
        print(f"[ERROR] Directory not found: {target}")
        sys.exit(1)
    # Language namespace directory (e.g. core/python) — recurse one level.
    # is_recipe_dir() already returns False for these, so we handle them
    # explicitly here before the generic validity check below.
    if target.name in LANGUAGE_NAMESPACE_DIRS:
        recipe_dirs = sorted(c for c in target.iterdir() if is_recipe_dir(c))
        if not recipe_dirs:
            print(f"[INFO] No recipe directories found under '{scope}/'.")
        return recipe_dirs
    if not is_recipe_dir(target):
        print(f"[ERROR] Not a valid recipe directory: {target}")
        sys.exit(1)
    return [target]


def _collect_root(root_name: str) -> list[Path]:
    """Return the recipe directories directly under a top-level root.

    Language namespace folders (e.g. core/python/) are recursed one level.
    """
    root_path = REPO_ROOT / root_name
    if not root_path.exists():
        print(f"[SKIP] '{root_name}/' does not exist.")
        return []

    recipe_dirs: list[Path] = []
    for p in sorted(root_path.iterdir()):
        if p.is_dir() and p.name in LANGUAGE_NAMESPACE_DIRS:
            recipe_dirs.extend(
                sorted(c for c in p.iterdir() if is_recipe_dir(c))
            )
        elif is_recipe_dir(p):
            recipe_dirs.append(p)

    if not recipe_dirs:
        print(f"[INFO] No recipe directories found under '{root_name}/'.")
    return recipe_dirs


def collect_recipe_dirs(scope: str | None) -> list[Path]:
    """Return the list of recipe directories to validate.

    scope can be:
      None / "all"              — all roots in RECIPE_ROOTS
      "core" / "contrib"        — a single top-level root
      "core/python"             — a language namespace dir (recurse one level)
      "core/some-recipe"        — a single flat recipe directory
      "core/python/some-recipe" — a single namespaced recipe directory
    """
    if scope is None or scope == "all":
        roots_to_scan = RECIPE_ROOTS
    elif scope in RECIPE_ROOTS:
        roots_to_scan = [scope]
    else:
        return _collect_scoped_path(scope)

    dirs: list[Path] = []
    for root_name in roots_to_scan:
        dirs.extend(_collect_root(root_name))
    return dirs


def main(scope: str | None = None) -> int:
    schema = load_schema()
    recipe_dirs = collect_recipe_dirs(scope)

    missing = []
    invalid = {}

    for recipe_dir in recipe_dirs:
        manifest_path = recipe_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            missing.append(str(recipe_dir.relative_to(REPO_ROOT)))
        else:
            errors = validate_manifest(manifest_path, schema)
            if errors:
                invalid[str(manifest_path.relative_to(REPO_ROOT))] = errors

    passed = True

    if missing:
        passed = False
        print(
            "\n[FAIL] Missing manifest.yaml in the following recipe"
            " directories:"
        )
        for d in missing:
            print(f"  - {d}/")
            # GitHub Actions annotation — surfaces in the PR Files tab
            print(
                f"::error file={d}/manifest.yaml::manifest.yaml is missing. "
                "Create one using the schema at "
                ".github/schemas/manifest-schema.json"
            )

    if invalid:
        passed = False
        print("\n[FAIL] Invalid manifest.yaml files:")
        for path, errors in invalid.items():
            print(f"\n  {path}:")
            for e in errors:
                print(f"    {e}")
            # Emit one annotation per file pointing at the manifest
            first_error = errors[0].strip()
            print(
                f"::error file={path}::{first_error} (+{len(errors) - 1} more)"
                if len(errors) > 1
                else f"::error file={path}::{first_error}"
            )

    if not passed:
        print(
            "\n========================================"
            "\n  ACTION REQUIRED: invalid manifest(s)"
            "\n========================================"
            "\n"
            "\nFix the manifest.yaml file(s) listed above, then push again."
            "\n"
            "\nReference:"
            "\n  Schema:  .github/schemas/manifest-schema.json"        )
        return 1

    checked = len(recipe_dirs)
    print(f"\n[PASS] All {checked} recipe manifest(s) are present and valid.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate recipe manifest.yaml files."
    )
    parser.add_argument(
        "scope",
        nargs="?",
        default=None,
        help=(
            "What to validate: 'all' (default), 'core', 'contrib', "
            "or a path to a single recipe (e.g. core/rag-agent-search)."
        ),
    )
    args = parser.parse_args()
    sys.exit(main(args.scope))
