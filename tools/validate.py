#!/usr/bin/env python3
"""
Entry point for all recipe validation tools.

Usage:
  uv run validate <subcommand> [recipe]

Subcommands:
  manifest  — check that manifest.yaml exists and conforms to the schema
  all       — run all available checks

Arguments:
  recipe    — optional path to a single recipe directory (e.g. core/rag-agent-search)
               if omitted, all recipes under core/ and contrib/ are checked

Examples:
  uv run validate manifest
  uv run validate manifest core/rag-agent-search
  uv run validate all
  uv run validate all core/rag-agent-search

Exit codes:
  0 — all checks passed
  1 — one or more checks failed
"""

import argparse
import sys

import validate_manifest

SUBCOMMANDS = {
    "manifest": ("Manifest validation", validate_manifest.main),
    # Register future tools here, e.g.:
    # "lint": ("Lint check", validate_lint.main),
}


def run_all(recipe: str | None) -> int:
    results = []
    for label, tool_main in SUBCOMMANDS.values():
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        exit_code = tool_main(recipe)
        results.append((label, exit_code))

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    all_passed = True
    for label, exit_code in results:
        status = "[PASS]" if exit_code == 0 else "[FAIL]"
        print(f"  {status} {label}")
        if exit_code != 0:
            all_passed = False

    return 0 if all_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="validate",
        description="Recipe validation tools.",
    )
    subcommands_help = ", ".join([*SUBCOMMANDS, "all"])
    parser.add_argument(
        "subcommand",
        choices=[*SUBCOMMANDS, "all"],
        metavar="subcommand",
        help=f"Validation to run: {subcommands_help}",
    )
    parser.add_argument(
        "recipe",
        nargs="?",
        default=None,
        help="Path to a single recipe directory (e.g. core/rag-agent-search). "
        "If omitted, all recipes are checked.",
    )
    args = parser.parse_args()

    if args.subcommand == "all":
        return run_all(args.recipe)

    _, tool_main = SUBCOMMANDS[args.subcommand]
    return tool_main(args.recipe)


if __name__ == "__main__":
    sys.exit(main())
