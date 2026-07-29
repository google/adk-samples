#!/usr/bin/env python3
"""
Validates README.md files in recipe directories under core/ and contrib/.

Checks:
  - Every recipe directory has a README.md file.
  - README.md is not empty.
  - README.md contains no TODO: placeholders.
  - README.md meets a minimum word count (description proxy).
  - README.md has a setup section (heading keyword match).
  - README.md has a run section (heading keyword match).
  - README.md has at least one fenced code block (run command proxy).

Usage:
  # Validate all recipes (both core/ and contrib/):
  python3 tools/validate_readme.py
  python3 tools/validate_readme.py all

  # Validate only core/ or contrib/:
  python3 tools/validate_readme.py core
  python3 tools/validate_readme.py contrib

  # Validate a single recipe:
  python3 tools/validate_readme.py core/python/rag-agent-search

  Dependencies are managed in pyproject.toml. Run `uv sync` once before using.

Exit codes:
  0 — all READMEs present and valid
  1 — one or more READMEs missing or invalid
"""

import argparse
import re
import sys
from pathlib import Path

import validate_manifest

REPO_ROOT = validate_manifest.REPO_ROOT
README_FILENAME = "README.md"
MIN_WORD_COUNT = 100

# Keywords checked against each heading (case-insensitive, word-boundary match).
# Broad enough to accept common synonyms without requiring exact names.
_SETUP_KEYWORDS = frozenset(
    [
        "setup",
        "prerequisites",
        "prerequisite",
        "installation",
        "install",
        "requirements",
        "requirement",
        "configuration",
        "getting started",
        "before you begin",
        "environment",
    ]
)

_RUN_KEYWORDS = frozenset(
    [
        "run",
        "running",
        "usage",
        "quickstart",
        "quick start",
        "start",
        "deploy",
        "deployment",
        "how to run",
        "how to use",
        "launch",
        "launching",
    ]
)

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def _has_section(content: str, keywords: frozenset[str]) -> bool:
    """Return True if any heading matches any keyword (word-boundary, case-insensitive)."""
    headings = [m.group(1).lower() for m in _HEADING_RE.finditer(content)]
    for heading in headings:
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, heading):
                return True
    return False


def validate_readme(readme_path: Path) -> list[str]:
    """Returns a list of error strings. Empty list means valid."""
    errors = []

    try:
        content = readme_path.read_text(encoding="utf-8")
    except OSError as e:
        errors.append(f"Cannot read README.md: {e}")
        return errors

    if not content.strip():
        errors.append("README.md is empty")
        return errors

    # No TODO: placeholders — catches unfilled scaffold or draft text.
    if re.search(r"TODO:", content):
        errors.append(
            "  [description] README.md contains TODO: placeholders — "
            "replace with real content before submitting."
        )

    # Minimum word count — proxy for a real description being present.
    word_count = len(content.split())
    if word_count < MIN_WORD_COUNT:
        errors.append(
            f"  [description] README.md is too short ({word_count} words, "
            f"minimum {MIN_WORD_COUNT}) — add a description, setup, and run "
            "section."
        )

    # Setup section — must have a heading matching a setup keyword.
    if not _has_section(content, _SETUP_KEYWORDS):
        errors.append(
            "  [setup] README.md is missing a setup section — add a heading "
            "containing one of: Setup, Prerequisites, Installation, "
            "Requirements, Configuration, Getting Started, Before You Begin."
        )

    # Run section — must have a heading matching a run keyword.
    if not _has_section(content, _RUN_KEYWORDS):
        errors.append(
            "  [run] README.md is missing a run section — add a heading "
            "containing one of: Run, Running, Usage, Quickstart, Start, "
            "Deploy, How to Run."
        )

    # At least one fenced code block — the run command must be shown.
    if "```" not in content:
        errors.append(
            "  [run] README.md has no fenced code block — show the exact "
            "command to start the agent in a ``` block."
        )

    return errors


def main(scope: str | None = None) -> int:
    recipe_dirs = validate_manifest.collect_recipe_dirs(scope)

    missing = []
    invalid = {}

    for recipe_dir in recipe_dirs:
        readme_path = recipe_dir / README_FILENAME
        if not readme_path.exists():
            missing.append(str(recipe_dir.relative_to(REPO_ROOT)))
        else:
            errors = validate_readme(readme_path)
            if errors:
                invalid[str(readme_path.relative_to(REPO_ROOT))] = errors

    passed = True

    if missing:
        passed = False
        print(
            "\n[FAIL] Missing README.md in the following recipe"
            " directories:"
        )
        for d in missing:
            print(f"  - {d}/")
            print(
                f"::error file={d}/README.md::README.md is missing. "
                "Add a README.md with a description, setup, and run section."
            )

    if invalid:
        passed = False
        print("\n[FAIL] Invalid README.md files:")
        for path, errors in invalid.items():
            print(f"\n  {path}:")
            for e in errors:
                print(f"    {e}")
            first_error = errors[0].strip()
            print(
                f"::error file={path}::{first_error} (+{len(errors) - 1} more)"
                if len(errors) > 1
                else f"::error file={path}::{first_error}"
            )

    if not passed:
        print(
            "\n========================================"
            "\n  ACTION REQUIRED: invalid README(s)"
            "\n========================================"
            "\n"
            "\nFix the README.md file(s) listed above, then push again."
            "\n"
            "\nReference:"
            "\n  Docs:  docs/recipe-handbook/anatomy.md"
        )
        return 1

    checked = len(recipe_dirs)
    print(f"\n[PASS] All {checked} recipe README(s) are present and valid.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate recipe README.md files."
    )
    parser.add_argument(
        "scope",
        nargs="?",
        default=None,
        help=(
            "What to validate: 'all' (default), 'core', 'contrib', "
            "or a path to a single recipe (e.g. core/python/rag-agent-search)."
        ),
    )
    args = parser.parse_args()
    sys.exit(main(args.scope))
