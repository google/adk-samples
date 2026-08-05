# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Consistency tests for .github/dependabot.yml.

That file is static and glob-based, so nothing regenerates it and no CI gate
compares it to the tree. Two things could therefore drift unnoticed:

  1. The globs stop covering directories that actually hold manifests — for
     example someone adds a recipe root, or narrows a pattern in a way that
     silently drops recursion. Recipes then receive no dependency updates,
     and nothing says so.
  2. The globs and recipe_manifests.SCAN_ROOTS disagree about which roots
     exist. That one is worse than it looks: the orphan cleanup treats
     anything outside SCAN_ROOTS as dead, so a root present in the config but
     missing from SCAN_ROOTS means its PRs get closed with --delete-branch.

These tests pin both. They cannot verify Dependabot's own glob implementation
— only a merge to the default branch can — so they check the patterns under
standard globstar semantics, which is what the GitHub documentation describes.
"""

from pathlib import Path

import pytest
import recipe_manifests as rm
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "dependabot.yml"


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def globbed_entries(config) -> list[dict]:
    """Entries that use `directories` (i.e. everything but github-actions)."""
    return [u for u in config["updates"] if "directories" in u]


def _segments_match(pattern: list[str], path: list[str]) -> bool:
    """Standard globstar matching over path segments.

    `**` matches zero or more segments; `*` matches exactly one, and never
    spans a separator.

    Implemented rather than delegated to Path.glob() because expanding
    `/core/**/*` against the working tree walks .venv and node_modules and
    takes minutes. Matching the scanner's output against the pattern is the
    same question asked in the cheap direction.
    """
    if not pattern:
        return not path
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        # Zero segments, or consume one and retry.
        if _segments_match(rest, path):
            return True
        return bool(path) and _segments_match(pattern, path[1:])
    if not path:
        return False
    if head != "*" and head != path[0]:
        return False
    return _segments_match(rest, path[1:])


def _matches_any(patterns: list[str], directory: str) -> bool:
    segs = directory.strip("/").split("/")
    return any(_segments_match(p.strip("/").split("/"), segs) for p in patterns)


# ---------------------------------------------------------------------------
# The matcher itself
#
# Every config test below depends on _segments_match. A matcher that returned
# True unconditionally would make all of them pass while proving nothing, so
# it is pinned first — negative cases included.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "directory", "expected"),
    [
        # The depth question the globs hinge on: does `/core/**/*` reach a
        # directory only one level below the root? Under standard globstar
        # `**` matches zero segments, so yes.
        ("/core/**/*", "/core/rag-vector-search", True),
        ("/core/**/*", "/core/python/deep-search", True),
        ("/core/**/*", "/core/python/rag/data_ingestion", True),
        ("/core/**/*", "/core/python/lhh/web/server", True),
        # The root itself holds no manifest and should not match.
        ("/core/**/*", "/core", False),
        # Other roots must not be caught by this pattern.
        ("/core/**/*", "/contrib/python/x", False),
        ("/core/**/*", "/skills/retail/x", False),
        # Retired roots must not match.
        ("/core/**/*", "/python/agents/legacy", False),
        # A single `*` never spans a separator.
        ("/core/*", "/core/python/deep-search", False),
        ("/core/*", "/core/rag-vector-search", True),
        # A directory that merely starts with the root name is not the root.
        ("/core/**/*", "/coreutils/x", False),
    ],
)
def test_segments_match(pattern, directory, expected):
    assert _matches_any([pattern], directory) is expected


def test_matcher_is_not_vacuously_true():
    assert not _matches_any(["/nope/**/*"], "/core/python/deep-search")
    assert not _matches_any([], "/core/python/deep-search")


# ---------------------------------------------------------------------------
# The config
# ---------------------------------------------------------------------------


def test_config_is_valid_yaml_with_version_2(config):
    assert config["version"] == 2
    assert config["updates"]


def test_every_detected_manifest_directory_is_covered_by_a_glob(config):
    """The core invariant: if the scanner can see a manifest there, the config
    must be telling Dependabot to look there."""
    by_eco = {u["package-ecosystem"]: u for u in config["updates"]}

    uncovered = []
    for ecosystem, directory in rm.scan():
        entry = by_eco.get(ecosystem)
        assert entry is not None, (
            f"{directory} has a {ecosystem} manifest but dependabot.yml "
            f"declares no {ecosystem} entry — it receives no updates"
        )
        if not _matches_any(entry["directories"], directory):
            uncovered.append((ecosystem, directory))

    assert not uncovered, (
        "these directories hold manifests but no glob matches them, so they "
        f"receive no dependency updates: {uncovered}"
    )


def test_globs_reach_every_depth_present_in_the_tree(config):
    """Recursion is load-bearing, not decorative. Manifests sit anywhere from
    one to four levels below a root (`core/rag-vector-search` up to
    `core/python/long-horizon-harness/web/server`), so a pattern that only
    matched a fixed depth would silently drop the rest."""
    depths = {d.strip("/").count("/") + 1 for _, d in rm.scan()}
    assert len(depths) > 1, (
        "expected manifests at several depths; if the tree flattened, this "
        "test no longer proves recursion is exercised"
    )

    by_eco = {u["package-ecosystem"]: u for u in config["updates"]}
    for ecosystem, directory in rm.scan():
        patterns = by_eco[ecosystem]["directories"]
        assert _matches_any(patterns, directory), (
            f"{directory} (depth {directory.strip('/').count('/') + 1}) is "
            f"not matched by {patterns}"
        )


def test_glob_roots_and_scan_roots_agree(globbed_entries):
    """A root in the config but not in SCAN_ROOTS is actively dangerous: the
    orphan cleanup treats anything outside SCAN_ROOTS as dead and closes its
    PRs with --delete-branch."""
    for entry in globbed_entries:
        roots = {p.strip("/").split("/")[0] for p in entry["directories"]}
        assert roots == set(rm.SCAN_ROOTS), (
            f"{entry['package-ecosystem']} globs {sorted(roots)} but "
            f"recipe_manifests.SCAN_ROOTS is {sorted(rm.SCAN_ROOTS)}"
        )


def test_retired_roots_are_not_globbed(globbed_entries):
    """Retired roots (`frozen_paths` in .github/policy.yml) are closed to new
    work. Globbing them would resurrect dependency updates for recipes the
    repo has deliberately stopped maintaining."""
    policy = yaml.safe_load(
        (rm.REPO_ROOT / ".github" / "policy.yml").read_text(encoding="utf-8")
    )
    retired = {p.split("/")[0] for p in policy["frozen_paths"]}
    for entry in globbed_entries:
        roots = {p.strip("/").split("/")[0] for p in entry["directories"]}
        assert not (roots & retired), (
            f"{entry['package-ecosystem']} globs retired root(s) "
            f"{sorted(roots & retired)}"
        )


def test_every_ecosystem_present_in_the_tree_has_a_config_entry(config):
    """The guard that lets dependabot.yml safely omit ecosystems it does not
    need yet.

    recipe_manifests.py detects five ecosystems; the config declares only
    those that have recipes, because a glob matching zero directories risks
    the whole file being rejected — which would stop every dependency update,
    security ones included.

    This test keeps that from becoming a silent gap: the first Go, Java, or
    Kotlin recipe fails here with an explicit instruction, rather than
    merging green and quietly receiving no updates.
    """
    configured = {u["package-ecosystem"] for u in config["updates"]}
    present = {eco for eco, _ in rm.scan()}
    missing = present - configured
    assert not missing, (
        f"{sorted(missing)} manifests exist in the tree but dependabot.yml "
        "declares no entry for them, so those recipes receive no dependency "
        "updates. Add a block for each, copying the shape of the `uv` entry."
    )


def test_github_actions_entry_uses_a_plain_directory(config):
    """GitHub documents `directory: "/"` for github-actions; it is not a glob,
    and Dependabot looks in /.github/workflows regardless."""
    entry = next(
        u
        for u in config["updates"]
        if u["package-ecosystem"] == "github-actions"
    )
    assert entry["directory"] == "/"
    assert "directories" not in entry


def test_github_actions_entry_declares_no_semver_cooldown(config):
    """github-actions releases (v1, v2, ...) are not semver. Dependabot
    rejects the WHOLE config if semver-major-days appears here, which would
    silently disable every ecosystem."""
    entry = next(
        u
        for u in config["updates"]
        if u["package-ecosystem"] == "github-actions"
    )
    assert "semver-major-days" not in entry.get("cooldown", {})


def test_no_entry_sets_group_by(globbed_entries):
    """Without `group-by`, Dependabot opens one PR per directory, matching the
    behaviour the previous per-directory config had. Setting
    `group-by: dependency-name` would collapse a bump across every recipe into
    a single repo-wide PR."""
    for entry in globbed_entries:
        for group in entry.get("groups", {}).values():
            assert "group-by" not in group


def test_open_pr_limits_leave_room_for_every_directory(config):
    """If the limit is enforced per entry rather than per directory, a cap
    below the directory count would throttle updates for whichever recipes
    happen to lose the race."""
    counts: dict[str, int] = {}
    for ecosystem, _ in rm.scan():
        counts[ecosystem] = counts.get(ecosystem, 0) + 1

    for entry in config["updates"]:
        eco = entry["package-ecosystem"]
        needed = counts.get(eco, 0)
        if needed:
            assert entry["open-pull-requests-limit"] >= needed, (
                f"{eco} covers {needed} directories but caps open PRs at "
                f"{entry['open-pull-requests-limit']}"
            )
