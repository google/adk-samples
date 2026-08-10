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

The repo does NOT configure version updates for recipe ecosystems. Recipe
owners own dependency freshness, and a monthly canary landing separately in
#2502 detects rot instead of Dependabot preventing it — the reasoning is in
the header of dependabot.yml.

That inverts what these tests used to guard. The old invariant was "every
ecosystem in the tree has a config entry", so that a new Go or Java recipe
failed loudly rather than silently going unmanaged. Under the current policy
the *presence* of such an entry is the thing worth catching, because it would
quietly restart the PR flood the policy exists to stop.

So the tests below pin the absence deliberately, and still pin the one entry
that remains. What they cannot do is verify Dependabot's own behaviour — only
a merge to the default branch can.
"""

from pathlib import Path

import close_orphan_dependabot_prs as orphans
import pytest
import recipe_manifests as rm
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "dependabot.yml"


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# Ecosystems recipe_manifests.py can detect in the tree. Every one of these
# is a RECIPE ecosystem, so none of them may appear in dependabot.yml while
# the current policy stands. Derived from the detector list rather than
# retyped, so a newly supported ecosystem is covered here the moment it is
# added there.
RECIPE_ECOSYSTEMS = {eco for eco, _ in rm.DETECTORS}


def test_config_is_valid_yaml_with_version_2(config):
    assert config["version"] == 2
    assert config["updates"]


def test_no_version_updates_for_recipe_ecosystems(config):
    """The core invariant of the current policy.

    Recipes own their own dependency freshness; the repo does not open
    version-update PRs against them. An entry here for a recipe ecosystem
    would restart that flood — 98 open PRs last time, none of them mergeable,
    with the review load landing on the CODEOWNERS catch-all rather than on
    the recipe's declared owner.

    This is not a ban. It is a tripwire: re-enabling version updates should be
    a deliberate change with a reviewer attached, so update this test in the
    same PR and say why.
    """
    configured = {u["package-ecosystem"] for u in config["updates"]}
    offenders = configured & RECIPE_ECOSYSTEMS
    assert not offenders, (
        f"dependabot.yml declares version updates for {sorted(offenders)}, "
        "which are recipe ecosystems. Recipe dependency freshness is the "
        "recipe owner's responsibility and rot is caught by the monthly "
        "recipe canary — see the header of dependabot.yml. If re-enabling "
        "this is intentional, update this test in the same change."
    )


def test_security_updates_are_not_what_this_file_controls():
    """Guard against the most likely misreading of the change above.

    Dependabot security updates are a repository setting; GitHub documents
    that dependabot.yml and security alerts do not interact. Someone could
    reasonably assume that removing the recipe entries also switched security
    PRs off and "restore" them for the wrong reason.

    There is nothing in the file to assert against, so this pins the
    explanation instead: the reasoning must stay written down where the next
    person will look for it.
    """
    text = CONFIG_PATH.read_text(encoding="utf-8").lower()
    assert "security" in text, (
        "dependabot.yml no longer explains that security updates are "
        "unaffected by this file. Removing recipe entries does not disable "
        "them; say so, or the next reader will re-add entries to 'restore' "
        "security coverage that was never lost."
    )


def test_every_entry_is_a_known_static_entry(config):
    """With recipe ecosystems gone, every remaining entry is a static one.

    A static entry is configured, never discovered, so
    close_orphan_dependabot_prs.py can only know about it by being told. One
    present here but missing from STATIC_ENTRIES has its PRs classified as
    orphans and closed with --delete-branch — and since such an entry
    produces roughly one grouped PR a week, the --max-close circuit breaker
    would never trip on it.
    """
    for entry in config["updates"]:
        assert "directories" not in entry, (
            f"{entry['package-ecosystem']} uses a `directories` glob. Globs "
            "existed to cover the recipe tree, which this file no longer "
            "manages; a remaining entry should target a fixed directory."
        )

    configured_static = {
        (u["package-ecosystem"], u["directory"])
        for u in config["updates"]
        if "directory" in u
    }
    known_static = set(rm.static_pairs())

    unknown = configured_static - known_static
    assert not unknown, (
        f"{sorted(unknown)} are configured in dependabot.yml but missing from "
        "recipe_manifests.STATIC_ENTRIES, so the orphan cleanup would close "
        "their PRs with --delete-branch. Add them there."
    )

    stale = known_static - configured_static
    assert not stale, (
        f"{sorted(stale)} are in recipe_manifests.STATIC_ENTRIES but not in "
        "dependabot.yml. Harmless, but the list is now describing an entry "
        "that does not exist — remove it."
    )


def test_orphan_cleanup_still_sees_the_recipe_tree(config):
    """Removing entries here must not make live recipes look abandoned.

    close_orphan_dependabot_prs.live_pairs() derives liveness from
    recipe_manifests.scan() — the tree — not from this file, which is what
    keeps the 98 PRs that were open when the policy changed from being swept
    up and closed with --delete-branch on the next housekeeping run. That
    indirection is easy to "simplify" away later, so pin it.

    Asserting that the live set reaches BEYOND what this file configures is
    what makes the pin real: a live_pairs() rewritten to read dependabot.yml
    could only ever return a subset of it, and this goes red.
    """
    configured = {
        orphans.BRANCH_PREFIX.get(eco, eco)
        for eco in (u["package-ecosystem"] for u in config["updates"])
    }
    tracked = {eco for eco, _ in orphans.live_pairs()}

    assert tracked - configured, (
        "close_orphan_dependabot_prs.live_pairs() no longer tracks any "
        "ecosystem beyond the ones dependabot.yml configures, which is what "
        "it looks like when liveness starts being read from this file "
        "instead of from the tree. The cleanup treats anything it cannot see "
        "as dead and closes it with --delete-branch."
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


def test_group_by_is_not_set_anywhere(config):
    """`group-by` at entry level is not valid config and would have Dependabot
    reject the whole file. The real home for the key is
    `groups.<name>.group-by`; both levels are checked so the test's scope
    matches its name.
    """
    for entry in config["updates"]:
        assert "group-by" not in entry, (
            f"{entry['package-ecosystem']}: `group-by` at entry level is not "
            "valid config and would have Dependabot reject the whole file"
        )
        for name, group in entry.get("groups", {}).items():
            assert "group-by" not in group, (
                f"{entry['package-ecosystem']}.groups.{name}: `group-by` "
                "changes how updates are batched into PRs. Nothing needs it "
                "while this file configures a single directory — add it "
                "deliberately or not at all."
            )
