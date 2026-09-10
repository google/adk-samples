#!/usr/bin/env python3
"""
Decide which recipes get a container image built, and what to call it.

Answers one question: which recipe directories under the live roots carry a
Dockerfile at their root, and what image path should each one publish to?

Why this exists as its own module
---------------------------------
`.agents/skills/make-python-recipe-deployable` writes the serving files a
recipe needs — Dockerfile, fast_api_app.py, app_utils/ — and then stops. Its
own words: "Image builds happen later via Cloud Build → Artifact Registry;
this skill's job ends when the files are correct." This module is the front
half of that later step: it turns the tree into a build matrix.

Recipe-root Dockerfiles only
----------------------------
A recipe can contain more than one Dockerfile. contrib/python/multiformat-
hybrid-rag ships three — one for the recipe, two for data-ingestion
sub-services it stands up. Only the file at the recipe root serves the agent
and satisfies the Agent Engine container contract; the others are backing
infrastructure with their own lifecycles. Publishing those under a recipe
image name would misrepresent what the image is, so the walk stops at the
first Dockerfile it finds down each path.

A Dockerfile is not by itself evidence of a recipe
--------------------------------------------------
skills/retail/virtual-tryon/assets/export-template carries one, and it is a
build asset shipped by a skill rather than an agent anybody deploys. The
manifest is what makes a directory a recipe — the schema at
.github/schemas/manifest-schema.json is the same thing `deployable` is
declared in — so a root Dockerfile earns an image only when a manifest.yaml
sits beside it. Without that test the walk publishes whatever happens to be
containerized, and derives a nonsense language ("retail") from the path while
doing it.

Live roots only
---------------
SCAN_ROOTS and SKIP_DIRS are imported from recipe_manifests rather than
restated, because the retired roots (python/agents, java/agents, ... — see
`frozen_paths` in .github/policy.yml) still contain Dockerfiles. Those paths
are closed to new work; images built from them would be published from code
nobody is allowed to fix. Importing means this module cannot drift out of
agreement with the rest of the tooling about what "live" means.

Image naming: <language>/<recipe>, not <root>/<language>/<recipe>
-----------------------------------------------------------------
The root is deliberately dropped. A recipe promoted from contrib/ to core/ is
the same recipe, and its published image name should not change underneath
consumers who have pinned it. Language is kept because it disambiguates:
core/kotlin/llm-auditor and contrib/python/llm-auditor coexist today, and the
leaf name alone would collide.

Zero third-party dependencies, matching its callers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from recipe_manifests import REPO_ROOT, SCAN_ROOTS, SKIP_DIRS

# Every image lands here. The repository is public-read (allUsers has
# roles/artifactregistry.reader), which is what lets Agent Engine pull an
# image into a customer tenant without a per-consumer IAM grant.
REGISTRY = "us-west1-docker.pkg.dev/adk-samples-repo-gcp-support/adk-recipes-registry"


def _language_and_name(recipe: Path) -> tuple[str, str]:
    """Split a recipe path into (language, name).

    Recipes live at <root>/<language>/<name>. Anything shallower than that has
    no language segment to report, so the root stands in for it.
    """
    parts = recipe.parts
    if len(parts) >= 3:
        return parts[1], parts[-1]
    return parts[0], parts[-1]


def discover(repo_root: Path) -> list[dict[str, str]]:
    """Every live recipe with a Dockerfile at its root, sorted by path."""
    found: list[dict[str, str]] = []
    for root in SCAN_ROOTS:
        base = repo_root / root
        if not base.is_dir():
            continue
        for dockerfile in sorted(base.rglob("Dockerfile")):
            recipe = dockerfile.parent
            rel = recipe.relative_to(repo_root)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if not (recipe / "manifest.yaml").is_file():
                continue
            # A Dockerfile nested under another recipe belongs to a
            # sub-service, not to a recipe of its own.
            if any(entry["path"] != str(rel) and str(rel).startswith(entry["path"] + "/")
                   for entry in found):
                continue
            language, name = _language_and_name(rel)
            found.append(
                {
                    "path": str(rel),
                    "name": name,
                    "language": language,
                    "image": f"{REGISTRY}/{language}/{name}",
                }
            )
    return found


def _changed_paths(ref: str) -> set[str]:
    """Files touched between `ref` and HEAD, as repo-relative paths."""
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{ref}...HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        # A missing or unrelated ref is not worth failing the build over:
        # falling back to "everything changed" errs toward rebuilding, which
        # is wasteful but never publishes a stale image.
        print(f"warning: git diff against {ref} failed, treating all recipes as changed",
              file=sys.stderr)
        return set()
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--changed-from",
        metavar="REF",
        help="Only include recipes with files touched since REF. Omit to include all.",
    )
    parser.add_argument(
        "--recipe",
        metavar="PATH",
        help="Limit to one recipe path (e.g. core/python/ambient-expense-agent).",
    )
    args = parser.parse_args()

    recipes = discover(REPO_ROOT)

    if args.recipe:
        wanted = args.recipe.rstrip("/")
        recipes = [r for r in recipes if r["path"] == wanted]
        if not recipes:
            print(f"error: no recipe with a root Dockerfile at {wanted!r}", file=sys.stderr)
            return 1

    if args.changed_from:
        changed = _changed_paths(args.changed_from)
        if changed:
            recipes = [r for r in recipes
                       if any(f.startswith(r["path"] + "/") for f in changed)]

    print(json.dumps({"include": recipes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
