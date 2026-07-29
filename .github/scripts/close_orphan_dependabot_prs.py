#!/usr/bin/env python3
"""
Close orphaned Dependabot PRs.

Parses .github/dependabot.yml to build the set of tracked (ecosystem, directory)
pairs, lists every open Dependabot PR, and closes any whose head-ref maps to
a pair that is NOT in that set. This catches PRs stranded when a recipe (and
therefore its dependabot.yml entry) is removed or renamed — Dependabot itself
does not close such PRs, so without this cleanup they linger indefinitely.

Uses `gh pr close <n> --delete-branch` with no explanatory comment: the GitHub
GraphQL `addComment` mutation has an anti-abuse throttle that trips on large
batches (observed in practice at ~80 comments in a burst). The close+delete-
branch itself is the audit signal; the workflow log below enumerates every
closed PR.

Invoked by .github/workflows/sync-dependabot-config.yml on every sync-workflow
run (belt-and-braces cleanup — cheap when there are no orphans).

Requires: `gh` on PATH, GITHUB_TOKEN in the environment.

Usage
-----
  python .github/scripts/close_orphan_dependabot_prs.py           # apply
  python .github/scripts/close_orphan_dependabot_prs.py --dry-run # list only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEPENDABOT_YML = REPO_ROOT / ".github" / "dependabot.yml"

# Dependabot renames a few ecosystems in the branch prefix relative to the
# `package-ecosystem` value used in dependabot.yml. Everything not listed here
# uses the same string in both places.
BRANCH_PREFIX = {
    "npm": "npm_and_yarn",
    "github-actions": "github_actions",
}

# The GitHub repo to operate on. Set by Actions automatically; fall back to
# the known upstream when running locally.
REPO = os.environ.get("GITHUB_REPOSITORY", "google/adk-samples")


def parse_tracked_pairs(yml_path: Path) -> set[tuple[str, str]]:
    """Return the set of (branch-prefix-ecosystem, directory) pairs tracked
    by the current dependabot.yml.

    The YAML is auto-generated with a highly regular shape:

        - package-ecosystem: "<eco>"
          directory: "<dir>"

    so a line-scan is sufficient (and keeps this script zero-dependency, like
    its sibling generate_dependabot.py).
    """
    pairs: set[tuple[str, str]] = set()
    current_eco: str | None = None
    for line in yml_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r'\s*-\s*package-ecosystem:\s*"([^"]+)"', line)
        if m:
            current_eco = m.group(1)
            continue
        m = re.match(r'\s*directory:\s*"([^"]+)"', line)
        if m and current_eco is not None:
            directory = m.group(1)
            prefix = BRANCH_PREFIX.get(current_eco, current_eco)
            pairs.add((prefix, directory))
            current_eco = None
    return pairs


def head_ref_matches(head_ref: str, pair: tuple[str, str]) -> bool:
    """True iff `head_ref` is a Dependabot branch targeting `pair`.

    Branch naming convention:
      - Root directory ("/"):  dependabot/<eco>/<package>
      - Non-root:              dependabot/<eco>/<dir-no-leading-slash>/<package>

    The tail after the directory prefix must be a single path segment (the
    package identifier), so we require no further slashes. That prevents a
    tracked pair for "/x" from spuriously claiming a branch that actually
    belongs to the untracked subdirectory "/x/y".
    """
    eco, directory = pair
    if directory == "/":
        prefix = f"dependabot/{eco}/"
    else:
        prefix = f"dependabot/{eco}/{directory.lstrip('/')}/"
    if not head_ref.startswith(prefix):
        return False
    tail = head_ref[len(prefix) :]
    return bool(tail) and "/" not in tail


def is_orphan(head_ref: str, tracked: set[tuple[str, str]]) -> bool:
    return not any(head_ref_matches(head_ref, p) for p in tracked)


def list_open_dependabot_prs() -> list[dict]:
    """Return [{number, headRefName}, ...] for every open Dependabot PR."""
    result = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            REPO,
            "--author",
            "app/dependabot",
            "--state",
            "open",
            "--limit",
            "500",
            "--json",
            "number,headRefName",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def close_pr(number: int) -> tuple[bool, str]:
    result = subprocess.run(
        [
            "gh",
            "pr",
            "close",
            str(number),
            "--repo",
            REPO,
            "--delete-branch",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, (result.stderr or result.stdout).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Close orphaned Dependabot PRs (see module docstring).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List orphans without closing them.",
    )
    args = parser.parse_args(argv)

    tracked = parse_tracked_pairs(DEPENDABOT_YML)
    print(f"dependabot.yml tracks {len(tracked)} (ecosystem, directory) pairs.")

    prs = list_open_dependabot_prs()
    print(f"Found {len(prs)} open Dependabot PRs on {REPO}.")

    orphans = [pr for pr in prs if is_orphan(pr["headRefName"], tracked)]
    print(f"Of those, {len(orphans)} target directories no longer in config.")

    if not orphans:
        return 0

    if args.dry_run:
        print("\n[dry-run] Would close:")
        for pr in orphans:
            print(f"  #{pr['number']}  {pr['headRefName']}")
        return 0

    print("\nClosing orphans (gh pr close --delete-branch, no comment):")
    failures: list[int] = []
    for pr in orphans:
        ok, err = close_pr(pr["number"])
        if ok:
            print(f"  closed #{pr['number']}  {pr['headRefName']}")
        else:
            print(
                f"  FAILED #{pr['number']}  {pr['headRefName']}  -- {err[:200]}",
                file=sys.stderr,
            )
            failures.append(pr["number"])

    if failures:
        print(
            f"\n{len(failures)} orphan close(s) failed: {failures}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
