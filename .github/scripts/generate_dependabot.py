#!/usr/bin/env python3
"""
Auto-generates .github/dependabot.yml by scanning core/, contrib/, and skills/
for language-specific manifest files across all 5 ADK languages.

Supported ecosystems
--------------------
  uv      Python  — directory contains uv.lock
  gomod   Go      — directory contains go.mod
  maven   Java    — directory contains pom.xml
  gradle  Kotlin  — directory contains both build.gradle.kts AND
                    settings.gradle.kts (the latter marks the root project;
                    sub-modules have only build.gradle.kts and are skipped)
  npm     TS/JS   — directory contains package.json with a "dependencies"
                    or "devDependencies" key (config-only package.json skipped)

Usage
-----
  python .github/scripts/generate_dependabot.py        # writes in-place
  python .github/scripts/generate_dependabot.py --check # exit 1 if stale

Invoked automatically by .github/workflows/sync-dependabot-config.yml on
every push to main that touches core/, contrib/, or skills/.

DO NOT edit .github/dependabot.yml by hand — this script owns it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = REPO_ROOT / ".github" / "dependabot.yml"

# Top-level directories to scan
SCAN_ROOTS = ["core", "contrib", "skills"]

# Directory names that are never recipe roots — skip entirely during the walk
SKIP_DIRS = {
    ".venv", "node_modules", ".gradle", ".git",
    "__pycache__", ".tox", ".mypy_cache", "dist", "build",
}

# ---------------------------------------------------------------------------
# Ecosystem detectors
# Each returns True if the given directory should get a Dependabot entry.
# ---------------------------------------------------------------------------


def _is_uv(d: Path) -> bool:
    return (d / "uv.lock").is_file()


def _is_gomod(d: Path) -> bool:
    return (d / "go.mod").is_file()


def _is_maven(d: Path) -> bool:
    return (d / "pom.xml").is_file()


def _is_gradle(d: Path) -> bool:
    # Only the root project has settings.gradle.kts; sub-modules do not.
    return (d / "build.gradle.kts").is_file() and (d / "settings.gradle.kts").is_file()


def _is_npm(d: Path) -> bool:
    pkg = d / "package.json"
    if not pkg.is_file():
        return False
    try:
        data = json.loads(pkg.read_text(encoding="utf-8"))
        return bool(data.get("dependencies") or data.get("devDependencies"))
    except (json.JSONDecodeError, OSError):
        return False


DETECTORS: list[tuple[str, object]] = [
    ("uv",     _is_uv),
    ("gomod",  _is_gomod),
    ("maven",  _is_maven),
    ("gradle", _is_gradle),
    ("npm",    _is_npm),
]

# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def scan(repo_root: Path) -> list[dict]:
    """Return one entry dict per detected (ecosystem, directory) pair."""
    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for root_name in SCAN_ROOTS:
        root_path = repo_root / root_name
        if not root_path.is_dir():
            continue

        for dirpath, dirnames, _ in os.walk(root_path):
            # Prune unwanted directories in-place so os.walk won't descend
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)

            directory = Path(dirpath)

            for ecosystem, detector in DETECTORS:
                if not detector(directory):  # type: ignore[operator]
                    continue

                rel = "/" + directory.relative_to(repo_root).as_posix()
                key = (ecosystem, rel)
                if key in seen:
                    continue
                seen.add(key)

                entries.append({"package-ecosystem": ecosystem, "directory": rel})

    entries.sort(key=lambda e: (e["package-ecosystem"], e["directory"]))
    return entries

# ---------------------------------------------------------------------------
# YAML renderer
# Keeps zero runtime dependencies — uses only stdlib.
# ---------------------------------------------------------------------------


def _render_entry(ecosystem: str, directory: str, *, extra_labels: list[str] | None = None) -> str:
    labels_lines = "      - dependencies"
    if extra_labels:
        labels_lines += "\n" + "\n".join(f"      - {lb}" for lb in extra_labels)

    group_name = "all-actions" if ecosystem == "github-actions" else "all-dependencies"

    # `semver-major-days` is only valid for ecosystems that follow semver.
    # github-actions releases (v1, v2, ...) don't, and Dependabot rejects the
    # whole config if it appears there.
    if ecosystem == "github-actions":
        cooldown_lines = "      default-days: 7"
    else:
        cooldown_lines = "      default-days: 7\n      semver-major-days: 14"

    return f"""\
  - package-ecosystem: "{ecosystem}"
    directory: "{directory}"
    schedule:
      interval: weekly
      day: monday
      time: "02:00"
    commit-message:
      prefix: chore
      include: scope
    labels:
{labels_lines}
    open-pull-requests-limit: 1
    # Wait a few days after a release before opening a PR, so the community
    # catches broken releases before we auto-merge them. Security updates
    # bypass this cooldown and fire immediately.
    cooldown:
{cooldown_lines}
    groups:
      {group_name}:
        patterns:
          - "*"
"""


def render(entries: list[dict]) -> str:
    header = """\
# AUTO-GENERATED — do not edit by hand.
# Owned by .github/scripts/generate_dependabot.py
#
# This file is regenerated automatically on every push to main that touches
# core/, contrib/, or skills/.  To force an immediate refresh, trigger the
# "Sync Dependabot Config" workflow manually from the Actions tab, or run:
#
#   python .github/scripts/generate_dependabot.py
#
# Ecosystems tracked: uv (Python), gomod (Go), maven (Java),
#                     gradle (Kotlin), npm (TypeScript/JS), github-actions

version: 2
updates:
"""
    body_parts = [
        _render_entry(e["package-ecosystem"], e["directory"])
        for e in entries
    ]
    # Static github-actions entry is always last
    body_parts.append(
        _render_entry("github-actions", "/", extra_labels=["github-actions"])
    )
    return header + "\n".join(body_parts)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate .github/dependabot.yml")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if dependabot.yml is stale (CI mode); do not write.",
    )
    args = parser.parse_args(argv)

    entries = scan(REPO_ROOT)
    content = render(entries)

    current = OUTPUT_FILE.read_text(encoding="utf-8") if OUTPUT_FILE.is_file() else ""

    if content == current:
        print(f"dependabot.yml is up to date ({len(entries)} recipe entries).")
        return 0

    if args.check:
        print("dependabot.yml is STALE. Run the script locally to regenerate:")
        print("  python .github/scripts/generate_dependabot.py")
        return 1

    OUTPUT_FILE.write_text(content, encoding="utf-8")
    print(f"dependabot.yml written: {len(entries)} recipe entries + github-actions.")
    for e in entries:
        print(f"  [{e['package-ecosystem']:8s}] {e['directory']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
