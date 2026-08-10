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
"""Unit tests for canary_issues.py.

This script @-mentions people and proposes deleting recipes, so its failure
modes are social as well as technical: a stage that fires twice is monthly
nagging, a stage that fires early is a deletion warning nobody had a chance to
act on, and treating an infrastructure blip as a recipe failure teaches
everyone to ignore the channel.

Every test below pins one of those.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import canary_issues as m
import pytest

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _issue(days_old: int, labels: set[str] | None = None) -> dict:
    return {
        "number": 42,
        "title": "t",
        "createdAt": (NOW - timedelta(days=days_old)).isoformat(),
        "labelNames": labels or set(),
    }


@pytest.fixture
def calls(monkeypatch):
    """Record what the script would do, without touching GitHub."""
    recorded: list[tuple] = []
    monkeypatch.setattr(
        m, "comment", lambda n, b, d: recorded.append(("comment", n, b))
    )
    monkeypatch.setattr(
        m, "add_label", lambda n, lbl, d: recorded.append(("label", n, lbl))
    )
    monkeypatch.setattr(m, "read_owner", lambda r, repo_root=None: "owner-x")
    return recorded


# ---------------------------------------------------------------------------
# Which recipes are considered failing
# ---------------------------------------------------------------------------


def test_infra_outcomes_never_count_as_a_recipe_failure():
    """A registry timeout is not the owner's fault. Filing against them for it
    is how a notification channel loses credibility."""
    results = [
        {"recipe": "a", "python": "3.11", "outcome": "infra"},
        {"recipe": "b", "python": "3.11", "outcome": "fail"},
        {"recipe": "c", "python": "3.11", "outcome": "pass"},
    ]
    assert set(m.failing_recipes(results)) == {"b"}


def test_one_failing_version_fails_the_recipe():
    results = [
        {"recipe": "a", "python": "3.11", "outcome": "pass"},
        {"recipe": "a", "python": "3.13", "outcome": "fail"},
    ]
    assert set(m.failing_recipes(results)) == {"a"}


# ---------------------------------------------------------------------------
# The escalation clock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("age", "expected_labels"),
    [
        (0, []),
        (29, []),
        (30, ["recipe:reminded"]),
        (59, ["recipe:reminded"]),
        (60, ["recipe:reminded", m.LABEL_ROTTING]),
        (90, ["recipe:reminded", m.LABEL_ROTTING, m.LABEL_DELETION]),
        (
            120,
            [
                "recipe:reminded",
                m.LABEL_ROTTING,
                m.LABEL_DELETION,
                "recipe:deletion-proposed",
            ],
        ),
    ],
)
def test_stages_fire_at_their_thresholds(calls, age, expected_labels):
    m.escalate("core/python/demo", _issue(age), dry=False, now=NOW)
    fired = [c[2] for c in calls if c[0] == "label"]
    assert fired == expected_labels


def test_a_stage_never_fires_twice(calls):
    """The label is the state. Without this the day-30 reminder would repeat
    every month — exactly the nagging the design exists to avoid."""
    already = {"recipe:reminded", m.LABEL_ROTTING}
    m.escalate("core/python/demo", _issue(65, already), dry=False, now=NOW)
    assert [c for c in calls if c[0] == "label"] == []
    assert [c for c in calls if c[0] == "comment"] == []


def test_a_late_run_catches_up_through_every_missed_stage(calls):
    """If the canary is down for months, a 100-day-old issue must not skip
    from day 0 straight to the deletion warning with no reminder in between."""
    m.escalate("core/python/demo", _issue(100), dry=False, now=NOW)
    fired = [c[2] for c in calls if c[0] == "label"]
    assert fired == ["recipe:reminded", m.LABEL_ROTTING, m.LABEL_DELETION]


def test_deletion_stages_tag_a_maintainer_not_only_the_owner(calls):
    """An unresponsive owner is the reason we got here, so the 90- and
    120-day notices have to reach someone who can decide."""
    m.escalate("core/python/demo", _issue(120), dry=False, now=NOW)
    bodies = [c[2] for c in calls if c[0] == "comment"]
    assert any(f"@{m.MAINTAINER}" in b for b in bodies[-2:])


# ---------------------------------------------------------------------------
# Honesty about what the canary can actually do
# ---------------------------------------------------------------------------


def test_without_an_elevated_token_the_canary_says_so(monkeypatch):
    """GITHUB_TOKEN cannot push to a code-owner-protected branch. Claiming a
    PR was opened when none was is worse than admitting the gap."""
    monkeypatch.delenv("CANARY_APP_TOKEN", raising=False)
    note = m._write_note("core/python/demo", "status: inactive")
    assert "by hand" in note
    assert "will not pretend" in note


def test_with_an_elevated_token_it_promises_a_pr(monkeypatch):
    monkeypatch.setenv("CANARY_APP_TOKEN", "x")
    note = m._write_note("core/python/demo", "status: inactive")
    assert "pull request" in note


# ---------------------------------------------------------------------------
# Owner routing
# ---------------------------------------------------------------------------


def _manifest(tmp_path: Path, body: str) -> Path:
    d = tmp_path / "core" / "python" / "demo"
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text(body, encoding="utf-8")
    return tmp_path


def test_reads_quoted_and_unquoted_poc(tmp_path):
    root = _manifest(
        tmp_path, 'status: active\nownership:\n  team: t\n  poc: "someone"\n'
    )
    assert m.read_owner("core/python/demo", root) == "someone"


def test_ignores_a_poc_outside_the_ownership_block(tmp_path):
    """`poc:` under a different key is not the owner. Mentioning whoever it
    names would notify an uninvolved person."""
    root = _manifest(
        tmp_path,
        "status: active\narchitecture:\n  poc: not-the-owner\n"
        "ownership:\n  team: t\n  poc: real-owner\n",
    )
    assert m.read_owner("core/python/demo", root) == "real-owner"


def test_missing_poc_falls_back_to_the_maintainer(tmp_path):
    root = _manifest(tmp_path, "status: active\nownership:\n  team: t\n")
    assert m.read_owner("core/python/demo", root) is None
    mention, note = m._mention(None)
    assert mention == f"@{m.MAINTAINER}"
    assert "no usable `ownership.poc`" in note


# ---------------------------------------------------------------------------
# The issue body
# ---------------------------------------------------------------------------


def test_body_names_the_passing_versions(monkeypatch):
    """The whole diagnostic value of a two-version matrix. An earlier version
    was handed only the failures and could never render this."""
    monkeypatch.setattr(m, "read_owner", lambda r, repo_root=None: "o")
    body = m.build_body(
        "core/python/demo",
        [
            {"recipe": "d", "python": "3.11", "outcome": "pass"},
            {
                "recipe": "d",
                "python": "3.13",
                "outcome": "fail",
                "step": "pytest",
                "detail": "boom",
            },
        ],
        "https://run",
    )
    assert "Python 3.13." in body
    assert "still passes on Python 3.11" in body
    # The table lists the failure and not the pass.
    assert "| 3.13 | pytest | boom |" in body
    assert "| 3.11 |" not in body


def test_body_states_this_is_not_a_dependency_bump_nag(monkeypatch):
    monkeypatch.setattr(m, "read_owner", lambda r, repo_root=None: "o")
    body = m.build_body(
        "core/python/demo",
        [{"recipe": "d", "python": "3.11", "outcome": "fail"}],
        "https://run",
    )
    assert "committed lockfile" in body
    assert "never opens version-bump PRs" in body


def test_title_is_stable_because_it_is_the_dedupe_key():
    """Anything varying in the title makes the canary fail to recognise its
    own issue and open a fresh one every month."""
    assert m.issue_title("core/python/x") == m.issue_title("core/python/x")
    assert "core/python/x" in m.issue_title("core/python/x")


# ---------------------------------------------------------------------------
# Fail-closed
# ---------------------------------------------------------------------------


def test_empty_results_refuse_to_act(tmp_path, capsys):
    """An empty result set and a run where every job died look identical.
    Acting would close every open issue as 'recovered'."""
    results = tmp_path / "r.json"
    results.write_text("[]", encoding="utf-8")
    assert m.main(["--results", str(results)]) == 1
    assert "Refusing to act" in capsys.readouterr().err


def test_gh_failure_is_raised_not_swallowed(monkeypatch):
    """A canary that cannot file issues must fail its run rather than report
    success having told nobody."""
    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *a, **k: type(
            "R", (), {"returncode": 1, "stdout": "", "stderr": "nope"}
        )(),
    )
    with pytest.raises(m.GhError, match="nope"):
        m.gh("issue", "list")
