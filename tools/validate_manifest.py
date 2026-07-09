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

  Dependencies are managed via the 'tools' dependency group in pyproject.toml.
  Run `uv sync --group tools` once before using.

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

OWNERSHIP_TEAM_PLACEHOLDER = "YOUR TEAM NAME"
OWNERSHIP_POC_PLACEHOLDER = "your-github-id"


def is_recipe_dir(path: Path) -> bool:
    """A recipe directory is any immediate subdirectory that contains
    more than just a README.md."""
    if not path.is_dir() or path.name.startswith("."):
        return False
    children = [p for p in path.iterdir() if p.name != "README.md"]
    return len(children) > 0


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
                    f'  [ownership.team] is still set to the placeholder value '
                    f'"{OWNERSHIP_TEAM_PLACEHOLDER}". Please replace it with a real team name.'
                )
            if ownership.get("poc") == OWNERSHIP_POC_PLACEHOLDER:
                errors.append(
                    f'  [ownership.poc] is still set to the placeholder value '
                    f'"{OWNERSHIP_POC_PLACEHOLDER}". Please replace it with a real GitHub ID.'
                )

    return errors


def collect_recipe_dirs(scope: str | None) -> list[Path]:
    """Return the list of recipe directories to validate.

    scope can be:
      None / "all"         — all roots in RECIPE_ROOTS
      "core" / "contrib"   — a single top-level root
      "core/some-recipe"   — a single recipe directory
    """
    # Resolve scope roots to scan
    if scope is None or scope == "all":
        roots_to_scan = RECIPE_ROOTS
    elif scope in RECIPE_ROOTS:
        roots_to_scan = [scope]
    else:
        # Treat as a path to a single recipe directory
        target = REPO_ROOT / scope
        if not target.exists():
            print(f"[ERROR] Directory not found: {target}")
            sys.exit(1)
        if not is_recipe_dir(target):
            print(f"[ERROR] Not a valid recipe directory: {target}")
            sys.exit(1)
        return [target]

    dirs = []
    for root_name in roots_to_scan:
        root_path = REPO_ROOT / root_name
        if not root_path.exists():
            print(f"[SKIP] '{root_name}/' does not exist.")
            continue
        recipe_dirs = sorted(p for p in root_path.iterdir() if is_recipe_dir(p))
        if not recipe_dirs:
            print(f"[INFO] No recipe directories found under '{root_name}/'.")
            continue
        dirs.extend(recipe_dirs)
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

    if invalid:
        passed = False
        print("\n[FAIL] Invalid manifest.yaml files:")
        for path, errors in invalid.items():
            print(f"\n  {path}:")
            for e in errors:
                print(f"    {e}")

    if passed:
        checked = len(recipe_dirs)
        print(
            f"\n[PASS] All {checked} recipe manifest(s) are present and valid."
        )
        return 0

    return 1


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
