#!/usr/bin/env python3
"""
Locate dependency manifests across the recipe tree.

Answers one question: which directories under core/, contrib/, and skills/
contain a dependency manifest, and for which package ecosystem?

Why this exists as its own module
---------------------------------
`.github/dependabot.yml` no longer enumerates directories — it uses glob
patterns, so Dependabot discovers manifests itself and nothing has to be
regenerated when a recipe is added or removed.

But close_orphan_dependabot_prs.py still needs the concrete list, to decide
whether an open Dependabot PR targets a directory that still exists. It used
to recover that list by parsing the enumerated `directory:` keys out of
dependabot.yml. With glob patterns there is nothing to parse, so it scans the
tree instead — which is also strictly more accurate, since the tree is the
ground truth the globs are resolved against.

Zero third-party dependencies, matching its callers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Top-level directories that hold recipes. Retired roots (python/agents,
# java/agents, ... — see `frozen_paths` in .github/policy.yml) are
# deliberately absent: they are closed to new work and are not dependency-
# managed.
SCAN_ROOTS = ["core", "contrib", "skills"]

# Directory names that are never recipe roots — pruned during the walk.
SKIP_DIRS = {
    ".venv",
    "node_modules",
    ".gradle",
    ".git",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    "dist",
    "build",
}


# ---------------------------------------------------------------------------
# Ecosystem detectors
# Each returns True if the given directory should be dependency-managed.
# ---------------------------------------------------------------------------


def _is_uv(d: Path) -> bool:
    return (d / "uv.lock").is_file()


def _is_gomod(d: Path) -> bool:
    return (d / "go.mod").is_file()


def _is_maven(d: Path) -> bool:
    return (d / "pom.xml").is_file()


def _is_gradle(d: Path) -> bool:
    # Only the root project has settings.gradle.kts; sub-modules do not.
    return (d / "build.gradle.kts").is_file() and (
        d / "settings.gradle.kts"
    ).is_file()


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
    ("uv", _is_uv),
    ("gomod", _is_gomod),
    ("maven", _is_maven),
    ("gradle", _is_gradle),
    ("npm", _is_npm),
]


def scan(repo_root: Path | None = None) -> list[tuple[str, str]]:
    """Return sorted (ecosystem, "/repo/relative/dir") pairs, one per manifest.

    A single recipe can yield more than one pair — `core/python/
    rag-vector-search` and its nested `data_ingestion/` each carry a uv.lock,
    and `core/python/long-horizon-harness/web/server` is an npm manifest three
    levels below its recipe root. That is why this walks the whole tree rather
    than globbing a fixed depth.
    """
    root = REPO_ROOT if repo_root is None else repo_root
    pairs: set[tuple[str, str]] = set()

    for root_name in SCAN_ROOTS:
        root_path = root / root_name
        if not root_path.is_dir():
            continue

        for dirpath, dirnames, _ in os.walk(root_path):
            # Prune in-place so os.walk does not descend into them.
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            directory = Path(dirpath)

            for ecosystem, detector in DETECTORS:
                if detector(directory):  # type: ignore[operator]
                    rel = "/" + directory.relative_to(root).as_posix()
                    pairs.add((ecosystem, rel))

    return sorted(pairs)
