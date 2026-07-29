#!/usr/bin/env python3
"""
Validates the structural shell of every recipe under core/ and contrib/
(and, once it lands, skills/) — the checks that are determined by folder
location and the recipe's declared language, NOT by anything language-
specific.

Checks performed, in order, for each recipe:
  1. manifest.yaml presence.
  2. manifest.yaml schema (delegated to validate_manifest).
  3. Folder name matches ^[a-z][a-z-]*$ and length <= policy limit.
  4. Folder size (bytes) and file count vs .github/policy.yml tiers
     (core vs contrib, default vs manifest.large=true), applying
     excluded_paths as a union across every language section.
  5. Required files present — the UNION of policy.required_files.always,
     policy.required_files.by_root[<root>], and
     policy.required_files.by_language[<manifest.language>].
     manifest.yaml itself is NOT listed in policy.required_files; check 1
     is authoritative for it, and duplicating the rule would double-report.

Language detection: manifest.language is the ONLY source of truth. Path
convention (e.g. core/python/foo/) carries no meaning to this checker.
When manifest.yaml is missing, checks 2 and the language-derived slice of
check 5 are skipped for that recipe — the maintainer still sees the
folder-name, size, and always/by_root file-requirement failures in the
same run, so they can fix everything in one shot.

Usage:
  # Validate all recipes (both core/ and contrib/):
  uv run validate structure
  uv run validate structure all

  # Validate only core/ or contrib/:
  uv run validate structure core
  uv run validate structure contrib

  # Validate a single recipe:
  uv run validate structure core/rag-agent-search
  uv run validate structure core/python/some-recipe

Exit codes:
  0 — every checked recipe passed every applicable check
  1 — one or more recipes failed one or more checks
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import validate_manifest as vm
import yaml

REPO_ROOT = Path(__file__).parent.parent
POLICY_PATH = REPO_ROOT / ".github" / "policy.yml"

# Every folder that may hold recipes at the top level. Kept aligned with
# validate_manifest.RECIPE_ROOTS but extensible: adding a new root here
# (e.g. "skills") plus a matching `by_root:` entry in policy.yml is all
# it takes to bring the new root under structural validation.
RECIPE_ROOTS: list[str] = list(vm.RECIPE_ROOTS)

# Reused from the manifest tool so both checkers agree on what counts as
# a recipe directory (see is_recipe_dir there).
LANGUAGE_NAMESPACE_DIRS = vm.LANGUAGE_NAMESPACE_DIRS

# ---------------------------------------------------------------------------
# Regex for legal recipe folder names. Kept in code (not policy.yml)
# because it's shape-based, not a tunable number — same rationale as the
# comment in the recipe_naming section of policy.yml.
# ---------------------------------------------------------------------------
_FOLDER_NAME_RE = re.compile(r"^[a-z][a-z-]*$")


# ===========================================================================
# Policy loading
# ===========================================================================


def load_policy() -> dict:
    """Load .github/policy.yml as a plain dict. Raises on any I/O or YAML
    error — the caller has no useful recovery."""
    with open(POLICY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def required_files_for(
    policy: dict, root: str, language: str | None
) -> list[str]:
    """Union of `always`, `by_root[root]`, and `by_language[language]`
    from policy.required_files. Missing sections and missing keys degrade
    silently to an empty list so partial policy configs remain usable
    (e.g. `skills:` not yet defined)."""
    rf = policy.get("required_files") or {}
    files: list[str] = []
    files.extend(rf.get("always") or [])
    files.extend((rf.get("by_root") or {}).get(root) or [])
    if language:
        files.extend((rf.get("by_language") or {}).get(language) or [])
    # Preserve order but drop duplicates — a file could be listed under
    # multiple sources without meaning it's required twice.
    seen: set[str] = set()
    deduped: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


# ===========================================================================
# Root & language detection
# ===========================================================================


def recipe_root_of(recipe_dir: Path) -> str | None:
    """Top-level folder the recipe lives in ('core', 'contrib', …) —
    the first path component relative to REPO_ROOT. Returns None if
    the recipe is somehow outside REPO_ROOT (shouldn't happen in normal
    use but guards against surprising callers)."""
    try:
        rel = recipe_dir.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return None
    parts = rel.parts
    return parts[0] if parts else None


def detect_language(manifest_path: Path) -> str | None:
    """Return `manifest.language` as a lowercase string, or None if the
    manifest is missing, unreadable, or does not declare a language.
    Does NOT validate the value — schema validation is a separate check."""
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    lang = data.get("language")
    if not isinstance(lang, str):
        return None
    return lang.strip().lower() or None


def is_large_recipe(manifest_path: Path) -> bool:
    """Return True if manifest.large == true. Missing/invalid manifest
    is treated as `large: false` (the stricter default tier applies)."""
    if not manifest_path.is_file():
        return False
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError:
        return False
    return isinstance(data, dict) and data.get("large") is True


# ===========================================================================
# Individual checks (each returns list[str]; empty list means pass)
# ===========================================================================


def check_folder_name(recipe_dir: Path, max_length: int) -> list[str]:
    """Folder basename must match ^[a-z][a-z-]*$ and stay within the
    length limit. Two independent violations are reported independently
    so the maintainer sees both in one CI run."""
    name = recipe_dir.name
    errors: list[str] = []
    if not _FOLDER_NAME_RE.match(name):
        errors.append(
            f"Folder name '{name}' is invalid. Only lowercase letters and "
            "hyphens are allowed (must start with a letter; no underscores, "
            "digits, or uppercase)."
        )
    if len(name) > max_length:
        errors.append(
            f"Folder name '{name}' is {len(name)} characters long; the "
            f"maximum allowed is {max_length}."
        )
    return errors


def _iter_excluded_dirs_files(policy: dict) -> tuple[set[str], set[str]]:
    """Union of excluded_paths.<section>.{dirs,files} across every
    language section. Kept as sets for O(1) membership tests. Values
    are basename patterns (plain names or fnmatch globs)."""
    excluded_dirs: set[str] = set()
    excluded_files: set[str] = set()
    excluded = policy.get("excluded_paths") or {}
    for section in excluded.values():
        if not isinstance(section, dict):
            continue
        for d in section.get("dirs") or []:
            excluded_dirs.add(d)
        for f in section.get("files") or []:
            excluded_files.add(f)
    return excluded_dirs, excluded_files


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    """True if `name` matches any of the fnmatch patterns. Plain names
    (no glob metachars) still work because fnmatch treats them as
    literal matches."""
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _walk_counted_files(recipe_dir: Path, policy: dict) -> list[Path]:
    """Walk the recipe tree and return every file that COUNTS toward
    the size/count limits — i.e. everything not pruned by
    excluded_paths. Directory pruning is applied per basename, so a
    `build/` anywhere in the tree is excluded (mirrors the current
    bash `find -name X -prune` behaviour)."""
    excluded_dirs, excluded_files = _iter_excluded_dirs_files(policy)
    counted: list[Path] = []
    # Iterative walk so we can prune directories in place — pathlib's
    # rglob offers no prune hook, and os.walk gives us the mutable
    # `dirs` list we need.
    import os

    for dirpath, dirnames, filenames in os.walk(recipe_dir):
        # Prune excluded directories BEFORE descending into them.
        dirnames[:] = [
            d for d in dirnames if not _matches_any(d, excluded_dirs)
        ]
        for fname in filenames:
            if _matches_any(fname, excluded_files):
                continue
            counted.append(Path(dirpath) / fname)
    return counted


def check_size_and_count(
    recipe_dir: Path, root: str, policy: dict, manifest_path: Path
) -> list[str]:
    """Enforce the size (MB) and file-count tiers from policy.yml.
    Tier is chosen by `root` (core vs contrib) then by manifest.large.
    Recipes under roots not covered by recipe_size_limits (e.g. a
    future 'skills' root without limits) skip these checks."""
    limits_by_root = policy.get("recipe_size_limits") or {}
    root_limits = limits_by_root.get(root)
    if not isinstance(root_limits, dict):
        # No configured size limits for this root — nothing to enforce.
        return []

    is_large = is_large_recipe(manifest_path)
    tier = "large" if is_large else "default"
    tier_limits = root_limits.get(tier)

    # Fail loudly when `large: true` is set but the requested tier is not
    # defined for this root. Previously the code silently fell back to
    # `default` via a boolean chain, so the recipe author got no signal
    # that their opt-in was being ignored. Either add a `large:` block
    # under `recipe_size_limits.<root>` in policy.yml or drop `large:
    # true` from the manifest — the current combination is a bug.
    if tier_limits is None:
        if tier != "default":
            return [
                f"manifest.large=true, but recipe_size_limits.{root} in "
                ".github/policy.yml does not define a 'large' tier. Add a "
                f"'{root}.large' block in policy.yml with max_files / "
                "max_size_mb, or remove 'large: true' from the manifest."
            ]
        # No `default` tier and no `large` tier requested — nothing to
        # enforce, matches the no-root-limits case.
        return []

    max_files = tier_limits.get("max_files")
    max_size_mb = tier_limits.get("max_size_mb")

    counted = _walk_counted_files(recipe_dir, policy)
    file_count = len(counted)
    total_bytes = sum(p.stat().st_size for p in counted if p.is_file())
    total_kb = total_bytes // 1024

    errors: list[str] = []
    if max_size_mb is not None:
        max_bytes = int(max_size_mb) * 1024 * 1024
        if total_bytes > max_bytes:
            errors.append(
                f"Recipe folder exceeds {max_size_mb} MB "
                f"({total_kb} KB, excluding paths listed in "
                ".github/policy.yml). Remove large files or binaries."
            )
    if max_files is not None and file_count > int(max_files):
        errors.append(
            f"Recipe folder contains {file_count} files, exceeding the "
            f"limit of {max_files}."
        )
    return errors


def check_required_files(
    recipe_dir: Path, root: str, language: str | None, policy: dict
) -> list[str]:
    """Every path in the union list must exist as a file (not a dir).
    Paths are recipe-relative and may include subdirectories
    (e.g. tests/test_runnability.py)."""
    required = required_files_for(policy, root, language)
    errors: list[str] = []
    for rel in required:
        target = recipe_dir / rel
        if not target.is_file():
            errors.append(f"Required file '{rel}' is missing.")
    return errors


# ===========================================================================
# Per-recipe orchestration
# ===========================================================================


def validate_recipe(recipe_dir: Path, policy: dict, schema: dict) -> list[str]:
    """Run every applicable check on one recipe. Returns the list of
    error strings (empty list means everything passed). Errors are
    prefixed with the check name so the CI output stays parseable."""
    errors: list[str] = []

    root = recipe_root_of(recipe_dir)
    if root is None or root not in RECIPE_ROOTS:
        return [
            f"Recipe path must live under one of {RECIPE_ROOTS}; "
            f"got '{recipe_dir}'."
        ]

    # Check 1: manifest.yaml presence.
    manifest_path = recipe_dir / vm.MANIFEST_FILENAME
    manifest_present = manifest_path.is_file()
    if not manifest_present:
        errors.append(f"[manifest] {vm.MANIFEST_FILENAME} is missing.")
        # Continue with the checks that don't need the manifest —
        # folder-name and (best-effort) size — so the maintainer sees
        # every problem in one shot rather than fix-and-retry chains.

    # Check 2: manifest.yaml schema (only if the file exists — a missing
    # manifest is already reported by check 1).
    if manifest_present:
        for err in vm.validate_manifest(manifest_path, schema):
            errors.append(f"[manifest] {err.strip()}")

    # Check 3: folder name.
    naming = policy.get("recipe_naming") or {}
    max_length = int(naming.get("max_folder_name_length", 30))
    for err in check_folder_name(recipe_dir, max_length):
        errors.append(f"[folder-name] {err}")

    # Check 4: folder size + file count.
    for err in check_size_and_count(recipe_dir, root, policy, manifest_path):
        errors.append(f"[size] {err}")

    # Check 5: required files (union of always + by_root + by_language).
    # Language detection is intentionally decoupled from schema validation:
    # the schema check above already reports an invalid `language` value;
    # here we simply use whatever the manifest declares (or None) so this
    # check still runs even if the manifest has unrelated schema issues.
    language = detect_language(manifest_path) if manifest_present else None
    for err in check_required_files(recipe_dir, root, language, policy):
        errors.append(f"[required-files] {err}")

    return errors


# ===========================================================================
# CLI entry point (mirrors validate_manifest.main)
# ===========================================================================


def main(scope: str | None = None) -> int:
    policy = load_policy()
    schema = vm.load_schema()
    recipe_dirs = vm.collect_recipe_dirs(scope)

    failed: dict[str, list[str]] = {}
    for recipe_dir in recipe_dirs:
        errors = validate_recipe(recipe_dir, policy, schema)
        if errors:
            failed[str(recipe_dir.relative_to(REPO_ROOT))] = errors

    if failed:
        print("\n[FAIL] Recipes with structural problems:")
        for rel, errors in failed.items():
            print(f"\n  {rel}/:")
            for e in errors:
                print(f"    - {e}")
            # One GitHub annotation per recipe, pointing at the recipe root.
            first = errors[0]
            extra = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
            print(f"::error file={rel}::{first}{extra}")

        print(
            "\n========================================"
            "\n  ACTION REQUIRED: recipe structure invalid"
            "\n========================================"
            "\n"
            "\nFix the issue(s) listed above, then push again."
            "\n"
            "\nReference:"
            "\n  Policy:  .github/policy.yml (required_files, "
            "recipe_size_limits, recipe_naming)"
            "\n  Schema:  .github/schemas/manifest-schema.json"
        )
        return 1

    print(
        f"\n[PASS] All {len(recipe_dirs)} recipe(s) passed structural checks."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate recipe folder structure."
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
