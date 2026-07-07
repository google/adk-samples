#!/usr/bin/env python3
"""
Entry point for all recipe validation tools.

Usage:
  uv run validate [subcommand] [scope]

Both arguments are optional:

  subcommand  — which check(s) to run; one of: manifest, all (default: all)
  scope       — what to validate; one of:
                  all            (default) validate core/ and contrib/
                  core           validate core/ only
                  contrib        validate contrib/ only
                  core/<recipe>  validate a single recipe directory

When only one argument is given and it looks like a path (contains '/') or
is a known root ('core', 'contrib'), it is treated as the scope and all
checks are run.

Examples:
  uv run validate core/rag-agent-search   # run all checks on one recipe
  uv run validate core                    # run all checks on core/ only
  uv run validate manifest                # run manifest check on everything
  uv run validate manifest core           # run manifest check on core/ only
  uv run validate manifest core/rag-agent-search

Exit codes:
  0 — all checks passed
  1 — one or more checks failed
"""

import sys

import validate_manifest

SUBCOMMANDS = {
    "manifest": ("Manifest validation", validate_manifest.main),
    # Register future tools here, e.g.:
    # "lint": ("Lint check", validate_lint.main),
}

VALID_SUBCOMMANDS = [*SUBCOMMANDS, "all"]
RECIPE_ROOTS = validate_manifest.RECIPE_ROOTS


def run_all(scope: str | None) -> int:
    results = []
    for label, tool_main in SUBCOMMANDS.values():
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        exit_code = tool_main(scope)
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


def looks_like_scope(arg: str) -> bool:
    """Return True if arg looks like a scope rather than a subcommand."""
    return "/" in arg or arg in RECIPE_ROOTS or arg == "all"


def main() -> int:
    args = sys.argv[1:]

    # Strip --help / -h and let Python handle it manually so we can keep
    # the argument parsing simple without argparse.
    if "-h" in args or "--help" in args:
        print(__doc__)
        return 0

    subcommand = None
    scope = None

    if len(args) == 0:
        subcommand = "all"
        scope = None
    elif len(args) == 1:
        if looks_like_scope(args[0]):
            # e.g. "uv run validate core/rag-agent-search"
            subcommand = "all"
            scope = args[0]
        elif args[0] in VALID_SUBCOMMANDS:
            subcommand = args[0]
            scope = None
        else:
            print(
                f"[ERROR] '{args[0]}' is not a valid subcommand or scope.\n"
                f"  Run 'uv run validate --help' for usage."
            )
            return 1
    elif len(args) == 2:
        if args[0] not in VALID_SUBCOMMANDS:
            print(
                f"[ERROR] '{args[0]}' is not a valid subcommand.\n"
                f"  Valid subcommands: {', '.join(VALID_SUBCOMMANDS)}\n"
                f"  Run 'uv run validate --help' for usage."
            )
            return 1
        subcommand = args[0]
        scope = args[1]
    else:
        print(
            "[ERROR] Too many arguments.\n"
            "  Run 'uv run validate --help' for usage."
        )
        return 1

    if subcommand == "all":
        return run_all(scope)

    _, tool_main = SUBCOMMANDS[subcommand]
    return tool_main(scope)


if __name__ == "__main__":
    sys.exit(main())
