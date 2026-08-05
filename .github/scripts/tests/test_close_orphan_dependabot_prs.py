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
"""Unit tests for close_orphan_dependabot_prs.py.

This script closes pull requests with `--delete-branch`. A false positive is
destructive and not trivially recoverable, so the classification logic and the
fail-closed guard are pinned here.

The dangerous case these tests exist for: the script used to derive its live
set by parsing enumerated `directory:` keys out of .github/dependabot.yml.
Once that file moved to glob patterns there was nothing left to parse, and the
old parser would have returned an empty set — making every open Dependabot PR
look orphaned.
"""

import close_orphan_dependabot_prs as c
import pytest

# ---------------------------------------------------------------------------
# head_ref_matches
# ---------------------------------------------------------------------------


def test_grouped_update_branch_matches_its_directory():
    assert c.head_ref_matches(
        "dependabot/uv/core/python/deep-search/all-dependencies-abc123",
        ("uv", "/core/python/deep-search"),
    )


def test_branch_for_a_different_directory_does_not_match():
    assert not c.head_ref_matches(
        "dependabot/uv/core/python/deep-search/all-dependencies-abc123",
        ("uv", "/core/python/safety-plugins"),
    )


def test_parent_directory_does_not_claim_a_nested_directorys_branch():
    """A tracked "/x" must not absorb a branch belonging to "/x/y", or
    removing the nested recipe would never be noticed."""
    assert not c.head_ref_matches(
        "dependabot/uv/core/python/rag/data_ingestion/some-package-1.2.3",
        ("uv", "/core/python/rag"),
    )


def test_nested_directory_matches_its_own_branch():
    assert c.head_ref_matches(
        "dependabot/uv/core/python/rag/data_ingestion/some-package-1.2.3",
        ("uv", "/core/python/rag/data_ingestion"),
    )


def test_root_directory_branch_matches():
    assert c.head_ref_matches(
        "dependabot/github_actions/actions/checkout-5",
        ("github_actions", "/"),
    )


@pytest.mark.parametrize(
    ("head_ref", "pair"),
    [
        # Go module paths contain slashes.
        (
            "dependabot/go_modules/contrib/go/my-recipe/github.com/spf13/cobra-1.8.0",
            ("go_modules", "/contrib/go/my-recipe"),
        ),
        # npm scoped packages contain slashes.
        (
            "dependabot/npm_and_yarn/core/python/x/frontend/types/node-20.1.0",
            ("npm_and_yarn", "/core/python/x/frontend"),
        ),
        # github-actions refs are owner/repo.
        (
            "dependabot/github_actions/actions/checkout-5",
            ("github_actions", "/"),
        ),
    ],
)
def test_multi_segment_package_names_still_match(head_ref, pair):
    """Ecosystems whose package identifiers contain slashes drop the
    single-segment tail rule. Without this, every Go security-update PR is
    classified as an orphan and deleted."""
    assert c.head_ref_matches(head_ref, pair)


def test_bare_directory_prefix_with_no_package_tail_does_not_match():
    assert not c.head_ref_matches(
        "dependabot/uv/core/python/deep-search/",
        ("uv", "/core/python/deep-search"),
    )


def test_non_dependabot_branch_does_not_match():
    assert not c.head_ref_matches(
        "feature/uv/core/python/deep-search/all-dependencies-abc123",
        ("uv", "/core/python/deep-search"),
    )


# ---------------------------------------------------------------------------
# is_orphan
# ---------------------------------------------------------------------------


def test_is_orphan_is_false_when_any_pair_matches():
    tracked = {
        ("uv", "/core/python/alpha"),
        ("uv", "/core/python/beta"),
    }
    assert not c.is_orphan(
        "dependabot/uv/core/python/beta/all-dependencies-1", tracked
    )


def test_is_orphan_is_true_for_a_removed_recipe():
    tracked = {("uv", "/core/python/alpha")}
    assert c.is_orphan(
        "dependabot/uv/core/python/deleted/all-dependencies-1", tracked
    )


def test_everything_is_an_orphan_against_an_empty_set():
    """The precise failure mode the fail-closed guard in main() prevents."""
    assert c.is_orphan("dependabot/uv/core/python/alpha/all-deps-1", set())


# ---------------------------------------------------------------------------
# live_pairs
# ---------------------------------------------------------------------------


def test_live_pairs_translates_ecosystem_names_to_branch_prefixes(monkeypatch):
    """dependabot.yml says `npm` and `gomod`; branches say `npm_and_yarn` and
    `go_modules`. Getting this wrong deletes every PR for those ecosystems."""
    monkeypatch.setattr(
        c.recipe_manifests,
        "scan",
        lambda: [
            ("npm", "/core/python/x/frontend"),
            ("gomod", "/contrib/go/y"),
            ("uv", "/core/python/z"),
        ],
    )
    assert c.live_pairs() == {
        ("npm_and_yarn", "/core/python/x/frontend"),
        ("go_modules", "/contrib/go/y"),
        ("uv", "/core/python/z"),
        ("github_actions", "/"),
    }


def test_live_pairs_always_includes_the_static_github_actions_entry(
    monkeypatch,
):
    """It is configured permanently rather than discovered by scanning, so
    nothing in the tree would keep its PRs from looking orphaned."""
    monkeypatch.setattr(c.recipe_manifests, "scan", list)
    assert c.live_pairs() == {("github_actions", "/")}


def test_live_pairs_on_the_real_repo_covers_everything_the_scan_found():
    """Derived from the scan rather than naming a recipe.

    An earlier version asserted `/core/python/deep-search` by name, which
    would break if that recipe were ever renamed — and would break on an
    unrelated PR, because this suite's triggers (tools-tests.yml) do not
    include core/**, so the PR doing the rename would never run it.
    """
    pairs = c.live_pairs()
    for ecosystem, directory in c.recipe_manifests.scan():
        expected = (c.BRANCH_PREFIX.get(ecosystem, ecosystem), directory)
        assert expected in pairs, f"{expected} missing from the live set"
    assert pairs, "the real repo should yield a non-empty live set"


def test_live_pairs_covers_every_static_entry():
    """The half that can delete PRs. Static entries are configured, not
    discovered, so nothing in the tree keeps their PRs from looking orphaned.
    Reading them from recipe_manifests is what stops this set and the
    generator's output from drifting apart."""
    pairs = c.live_pairs()
    for ecosystem, directory in c.recipe_manifests.static_pairs():
        expected = (c.BRANCH_PREFIX.get(ecosystem, ecosystem), directory)
        assert expected in pairs, f"static entry {expected} not recognised"


def test_a_newly_added_static_entry_is_recognised_without_code_changes(
    monkeypatch,
):
    """Regression guard for the drift that used to be possible: an entry
    added to the generator alone had its PRs classified as orphans and closed
    with --delete-branch, at roughly one grouped PR a week — far too slow for
    --max-close to notice."""
    monkeypatch.setattr(
        c.recipe_manifests,
        "STATIC_ENTRIES",
        [*c.recipe_manifests.STATIC_ENTRIES, ("npm", "/tooling", ["tooling"])],
    )
    pairs = c.live_pairs()
    assert ("npm_and_yarn", "/tooling") in pairs
    assert not c.is_orphan(
        "dependabot/npm_and_yarn/tooling/all-dependencies-abc", pairs
    )


# ---------------------------------------------------------------------------
# main() fail-closed guard
# ---------------------------------------------------------------------------


def test_main_refuses_to_run_when_the_scan_finds_nothing(monkeypatch, capsys):
    """An empty scan means something is broken (wrong cwd, partial checkout),
    not that the repo has no recipes. Acting on it would close and delete the
    branch of every open Dependabot PR."""
    monkeypatch.setattr(c.recipe_manifests, "scan", list)

    def explode():
        raise AssertionError("must not reach the GitHub API")

    monkeypatch.setattr(c, "list_open_dependabot_prs", explode)

    assert c.main(["--dry-run"]) == 1
    assert "Refusing to close anything" in capsys.readouterr().err


def test_main_proceeds_when_the_scan_finds_recipes(monkeypatch, capsys):
    monkeypatch.setattr(
        c.recipe_manifests,
        "scan",
        lambda: [("uv", "/core/python/alpha")],
    )
    monkeypatch.setattr(
        c,
        "list_open_dependabot_prs",
        lambda: [
            {
                "number": 1,
                "headRefName": "dependabot/uv/core/python/alpha/all-deps-1",
            },
            {
                "number": 2,
                "headRefName": "dependabot/uv/core/python/gone/all-deps-2",
            },
        ],
    )

    assert c.main(["--dry-run"]) == 0
    # Compare whole tokens, not substrings: `"#1" in out` also matches "#10".
    listed = {
        ln.split()[0]
        for ln in capsys.readouterr().out.splitlines()
        if ln.startswith("  #")
    }
    assert listed == {"#2"}, "only the removed recipe's PR should be listed"


def test_main_refuses_when_the_orphan_batch_exceeds_max_close(
    monkeypatch, capsys
):
    """The empty-scan guard only catches a scan returning NOTHING. A scan
    returning a plausible-but-wrong set, or a change to how Dependabot names
    branches, would reclassify healthy PRs as orphans — and would surface as
    an unusually large batch rather than as an empty one."""
    monkeypatch.setattr(
        c.recipe_manifests, "scan", lambda: [("uv", "/core/python/alpha")]
    )
    monkeypatch.setattr(
        c,
        "list_open_dependabot_prs",
        lambda: [
            {"number": n, "headRefName": f"dependabot/uv/core/python/g{n}/a-1"}
            for n in range(10)
        ],
    )

    def explode(number):
        raise AssertionError(f"must not close #{number} past the limit")

    monkeypatch.setattr(c, "close_pr", explode)

    assert c.main(["--max-close", "5"]) == 1
    err = capsys.readouterr().err
    assert "exceeds the --max-close limit" in err
    # The operator needs the whole list to judge it, and the exact re-run
    # value so they are not left guessing.
    assert "--max-close 10" in err


def test_max_close_limit_is_inclusive(monkeypatch):
    """A batch exactly at the limit passes, so a documented limit of N means
    "N is fine" rather than "N-1 is fine"."""
    monkeypatch.setattr(
        c.recipe_manifests, "scan", lambda: [("uv", "/core/python/alpha")]
    )
    monkeypatch.setattr(
        c,
        "list_open_dependabot_prs",
        lambda: [
            {"number": n, "headRefName": f"dependabot/uv/core/python/g{n}/a-1"}
            for n in range(3)
        ],
    )
    assert c.main(["--dry-run", "--max-close", "3"]) == 0


def test_default_max_close_is_actually_applied(monkeypatch, capsys):
    """Behavioural, not a range assertion. `0 < DEFAULT_MAX_CLOSE < 100` held
    just as happily at 1 or 99, so it proved nothing about the default being
    wired to anything."""
    monkeypatch.setattr(c, "DEFAULT_MAX_CLOSE", 2)
    monkeypatch.setattr(
        c.recipe_manifests, "scan", lambda: [("uv", "/core/python/alpha")]
    )
    monkeypatch.setattr(
        c,
        "list_open_dependabot_prs",
        lambda: [
            {"number": n, "headRefName": f"dependabot/uv/core/python/g{n}/a-1"}
            for n in range(5)
        ],
    )
    monkeypatch.setattr(
        c, "close_pr", lambda n: pytest.fail(f"must not close #{n}")
    )

    # No --max-close flag: the parser default must be DEFAULT_MAX_CLOSE.
    assert c.main([]) == 1
    assert "limit of 2" in capsys.readouterr().err


def test_dry_run_is_never_blocked_by_the_close_limit(monkeypatch, capsys):
    """A dry run only reports, so capping it would make the escape hatch
    unreachable: the advice for an oversized batch is to review the list,
    and reviewing it is exactly what a dry run does."""
    monkeypatch.setattr(
        c.recipe_manifests, "scan", lambda: [("uv", "/core/python/alpha")]
    )
    monkeypatch.setattr(
        c,
        "list_open_dependabot_prs",
        lambda: [
            {"number": n, "headRefName": f"dependabot/uv/core/python/g{n}/a-1"}
            for n in range(10)
        ],
    )
    monkeypatch.setattr(
        c, "close_pr", lambda n: pytest.fail(f"dry run closed #{n}")
    )

    assert c.main(["--dry-run", "--max-close", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("dependabot/uv/core/python/g") == 10, "must list them all"
    # And still say a real run would refuse, with the value to use.
    assert "--max-close 10" in out


# ---------------------------------------------------------------------------
# The close loop
#
# Nothing exercised this path before: main()-never-closes and
# close-failures-ignored were both survivable mutations.
# ---------------------------------------------------------------------------


def test_every_orphan_is_closed_and_nothing_else_is(monkeypatch, capsys):
    monkeypatch.setattr(
        c.recipe_manifests, "scan", lambda: [("uv", "/core/python/alpha")]
    )
    monkeypatch.setattr(
        c,
        "list_open_dependabot_prs",
        lambda: [
            {"number": 1, "headRefName": "dependabot/uv/core/python/alpha/a-1"},
            {"number": 2, "headRefName": "dependabot/uv/core/python/gone/a-2"},
            {"number": 3, "headRefName": "dependabot/uv/core/python/dead/a-3"},
        ],
    )
    closed = []
    monkeypatch.setattr(
        c, "close_pr", lambda n: (closed.append(n), (True, ""))[1]
    )

    assert c.main([]) == 0
    assert closed == [2, 3], "the live recipe's PR must be left alone"
    assert "closed #2" in capsys.readouterr().out


def test_a_failed_close_is_reported_and_fails_the_run(monkeypatch, capsys):
    """A 403 or a race must not be swallowed — the job has to go red, or a
    branch silently survives and the next run tries again forever."""
    monkeypatch.setattr(
        c.recipe_manifests, "scan", lambda: [("uv", "/core/python/alpha")]
    )
    monkeypatch.setattr(
        c,
        "list_open_dependabot_prs",
        lambda: [
            {"number": 7, "headRefName": "dependabot/uv/core/python/gone/a-1"},
            {"number": 8, "headRefName": "dependabot/uv/core/python/dead/a-2"},
        ],
    )
    monkeypatch.setattr(
        c,
        "close_pr",
        lambda n: (True, "") if n == 7 else (False, "403 Forbidden"),
    )

    assert c.main([]) == 1
    captured = capsys.readouterr()
    assert "closed #7" in captured.out
    assert "FAILED #8" in captured.err
    assert "403 Forbidden" in captured.err


def test_a_successful_run_with_no_orphans_closes_nothing(monkeypatch):
    monkeypatch.setattr(
        c.recipe_manifests, "scan", lambda: [("uv", "/core/python/alpha")]
    )
    monkeypatch.setattr(
        c,
        "list_open_dependabot_prs",
        lambda: [
            {"number": 1, "headRefName": "dependabot/uv/core/python/alpha/a-1"}
        ],
    )
    monkeypatch.setattr(
        c, "close_pr", lambda n: pytest.fail(f"closed #{n} with no orphans")
    )
    assert c.main([]) == 0


def test_main_does_not_close_anything_in_dry_run(monkeypatch):
    monkeypatch.setattr(
        c.recipe_manifests,
        "scan",
        lambda: [("uv", "/core/python/alpha")],
    )
    monkeypatch.setattr(
        c,
        "list_open_dependabot_prs",
        lambda: [
            {"number": 9, "headRefName": "dependabot/uv/core/python/x/all-1"}
        ],
    )

    def explode(number):
        raise AssertionError(f"dry run must not close #{number}")

    monkeypatch.setattr(c, "close_pr", explode)
    assert c.main(["--dry-run"]) == 0
