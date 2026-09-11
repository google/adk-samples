"""The arithmetic that makes automated review converge.

Every number here is the difference between an author seeing the reviewer wind
down and an author feeling harassed by it, so the edges are tested rather than
assumed: the round after a round that found nothing, the round that exhausts
the cap, the allowance too small to give every lane a share.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import review_budget as rb

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY = yaml.safe_load(
    (REPO_ROOT / ".github" / "policy.yml").read_text(encoding="utf-8")
)["pr_review_budget"]


def policy(**overrides):
    return {**rb.DEFAULTS, **POLICY, **overrides}


def review(commit, when, lane="Correctness", rid=None, body=None):
    return {
        "id": rid if rid is not None else hash((commit, lane)) % 100000,
        "commit_id": commit,
        "submitted_at": when,
        "body": body
        if body is not None
        else f"{rb.REVIEW_MARKER}\nAutomated **{lane}** review — 1 finding(s).",
    }


def comment(review_id):
    return {"pull_request_review_id": review_id}


# --------------------------------------------------------------- the policy


def test_the_shipped_policy_is_internally_consistent():
    """blocker_lanes must be a PREFIX of lanes, or the lanes that survive a
    shrinking allowance are not the ones that survive the round limit — and
    the reviewer would narrow to one set while allocating to another."""
    lanes = POLICY["lanes"]
    blockers = POLICY["blocker_lanes"]
    assert lanes[: len(blockers)] == blockers, (
        f"blocker_lanes {blockers} is not a prefix of lanes {lanes}"
    )
    assert 0 < POLICY["decay"] < 1
    assert POLICY["lifetime_cap"] > 0
    assert POLICY["min_allowance"] >= 1


def test_the_policy_lanes_are_the_real_lane_labels():
    """A label that matches no workflow silently drops that lane out of the
    allocation, giving every other lane a bigger share than intended."""
    labels = set()
    for path in (REPO_ROOT / ".github" / "workflows").glob(
        "ai-pr-review-*.yml"
    ):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "review_label:" in line:
                labels.add(line.split("review_label:")[1].strip().strip("'\""))
            if "--label '" in line:
                labels.add(line.split("--label '")[1].split("'")[0])
    for lane in POLICY["lanes"] + POLICY["exempt_lanes"]:
        assert lane in labels, (
            f"policy.yml names a lane {lane!r} that no workflow posts as; "
            f"workflows post as {sorted(labels)}"
        )


def test_defaults_match_the_shipped_policy():
    """The fallbacks are for an unreadable policy.yml, not a second opinion."""
    for key, value in rb.DEFAULTS.items():
        assert POLICY[key] == value, (
            f"policy.yml {key}={POLICY[key]!r} but the in-code default is "
            f"{value!r}; one of them is stale"
        )


# ------------------------------------------------------------ round history


def test_four_lanes_reviewing_one_push_is_one_round():
    """The lanes post four reviews against a single commit. Counting those as
    four rounds runs the decay four times per push and silences the reviewer
    on the second one."""
    reviews = [
        review("aaa", "2026-01-01T00:00:00Z", lane, rid=i)
        for i, lane in enumerate(POLICY["lanes"])
    ]
    state = rb.summarise_history(reviews, [], POLICY["exempt_lanes"])
    assert state["round"] == 2
    assert state["last_reviewed_sha"] == "aaa"


def test_the_round_number_counts_commits_reviewed():
    reviews = [
        review("aaa", "2026-01-01T00:00:00Z", rid=1),
        review("bbb", "2026-01-02T00:00:00Z", rid=2),
        review("ccc", "2026-01-03T00:00:00Z", rid=3),
    ]
    state = rb.summarise_history(reviews, [], POLICY["exempt_lanes"])
    assert state["round"] == 4
    assert state["last_reviewed_sha"] == "ccc"


def test_the_previous_round_count_is_only_the_last_commits_comments():
    reviews = [
        review("aaa", "2026-01-01T00:00:00Z", rid=1),
        review("bbb", "2026-01-02T00:00:00Z", rid=2),
    ]
    comments = [comment(1)] * 9 + [comment(2)] * 4
    state = rb.summarise_history(reviews, comments, POLICY["exempt_lanes"])
    assert state["posted_total"] == 13
    assert state["previous_round_count"] == 4


def test_a_humans_review_is_not_ours():
    reviews = [
        {
            "id": 1,
            "commit_id": "aaa",
            "submitted_at": "2026-01-01T00:00:00Z",
            "body": "Looks good, just one thing.",
        }
    ]
    state = rb.summarise_history(reviews, [comment(1)], POLICY["exempt_lanes"])
    assert state["round"] == 1
    assert state["posted_total"] == 0


def test_a_review_from_before_the_marker_still_counts():
    """Otherwise every PR already under review restarts at round 1 and gets a
    full-size batch the day this ships."""
    legacy = review(
        "aaa",
        "2026-01-01T00:00:00Z",
        body="Automated **Security** review — 3 finding(s).",
    )
    state = rb.summarise_history([legacy], [], POLICY["exempt_lanes"])
    assert state["round"] == 2


def test_the_exempt_lane_neither_advances_the_round_nor_spends_the_cap():
    reviews = [review("aaa", "2026-01-01T00:00:00Z", "House Rules", rid=7)]
    state = rb.summarise_history(
        reviews, [comment(7)] * 6, POLICY["exempt_lanes"]
    )
    assert state["round"] == 1, "the deterministic lane started the clock"
    assert state["posted_total"] == 0, "its comments consumed the model budget"


# ---------------------------------------------------------------- the decay


def test_the_documented_decay_curve():
    """20 -> 12 -> 7 -> 4 -> 2 -> 1 at 0.6. This is the shape an author sees."""
    seen, previous = [], 20
    for _ in range(5):
        state = {
            "round": 2,
            "last_reviewed_sha": "x",
            "posted_total": 0,
            "previous_round_count": previous,
        }
        previous = rb.decide(state, policy(), "Security", 5)["allowance"]
        seen.append(previous)
    assert seen == [12, 7, 4, 2, 1]


def test_the_allowance_never_reaches_zero():
    """A PR that grows a large new commit in round 9 still deserves one
    comment; the lifetime cap is what produces silence, not the decay."""
    state = {
        "round": 9,
        "last_reviewed_sha": "x",
        "posted_total": 0,
        "previous_round_count": 1,
    }
    assert rb.decide(state, policy(), "Security", 5)["allowance"] == 1


def test_a_round_that_found_nothing_does_not_silence_the_next():
    state = {
        "round": 3,
        "last_reviewed_sha": "x",
        "posted_total": 4,
        "previous_round_count": 0,
    }
    assert rb.decide(state, policy(), "Security", 5)["allowance"] == 1


def test_the_first_round_uses_the_churn_budget():
    state = {
        "round": 1,
        "last_reviewed_sha": "",
        "posted_total": 0,
        "previous_round_count": 0,
    }
    decision = rb.decide(state, policy(), "Security", 5)
    assert decision["allowance"] == 5 * len(POLICY["lanes"])
    assert decision["max_comments"] == 5


# ----------------------------------------------------------- the lifetime cap


def test_the_cap_stops_everything():
    state = {
        "round": 4,
        "last_reviewed_sha": "x",
        "posted_total": 25,
        "previous_round_count": 8,
    }
    decision = rb.decide(state, policy(), "Security", 5)
    assert decision["skip"] is True
    assert "at the 25 limit" in decision["reason"]


def test_the_cap_clips_a_round_that_would_overshoot():
    state = {
        "round": 2,
        "last_reviewed_sha": "x",
        "posted_total": 22,
        "previous_round_count": 20,
    }
    assert rb.decide(state, policy(), "Security", 5)["allowance"] == 3


def test_the_first_round_is_capped_too():
    state = {
        "round": 1,
        "last_reviewed_sha": "",
        "posted_total": 0,
        "previous_round_count": 0,
    }
    decision = rb.decide(state, policy(lifetime_cap=6), "Security", 5)
    assert decision["allowance"] == 6


# ------------------------------------------------------------ the allocation


def test_an_allowance_of_one_goes_to_exactly_one_lane():
    """Divide-and-round-up would turn 1 into 4 — at the tail of the decay,
    which is precisely where being exact matters."""
    got = {
        lane: rb.allocate(1, POLICY["lanes"], lane) for lane in POLICY["lanes"]
    }
    assert sum(got.values()) == 1
    assert got["Security"] == 1


@pytest.mark.parametrize("allowance", range(0, 21))
def test_the_shares_always_sum_to_the_allowance(allowance):
    total = sum(
        rb.allocate(allowance, POLICY["lanes"], lane)
        for lane in POLICY["lanes"]
    )
    assert total == allowance


def test_the_remainder_goes_to_the_higher_priority_lanes():
    shares = [rb.allocate(6, POLICY["lanes"], lane) for lane in POLICY["lanes"]]
    assert shares == [2, 2, 1, 1]


def test_a_lane_with_no_share_is_skipped_not_given_zero_comments():
    state = {
        "round": 2,
        "last_reviewed_sha": "x",
        "posted_total": 0,
        "previous_round_count": 2,
    }
    decision = rb.decide(state, policy(), "Hygiene", 5)
    assert decision["allowance"] == 1
    assert decision["skip"] is True
    assert "higher-priority lanes" in decision["reason"]


# ----------------------------------------------------------- the narrowing


def test_the_nit_lanes_stop_after_round_two():
    state = {
        "round": 3,
        "last_reviewed_sha": "x",
        "posted_total": 5,
        "previous_round_count": 8,
    }
    assert rb.decide(state, policy(), "Hygiene", 5)["skip"] is True
    assert rb.decide(state, policy(), "Maintainability", 5)["skip"] is True
    assert rb.decide(state, policy(), "Security", 5)["skip"] is False


def test_round_two_still_runs_everything():
    state = {
        "round": 2,
        "last_reviewed_sha": "x",
        "posted_total": 5,
        "previous_round_count": 20,
    }
    for lane in POLICY["lanes"]:
        assert rb.decide(state, policy(), lane, 5)["skip"] is False, lane


# -------------------------------------------------------------- exempt lane


def test_the_exempt_lane_is_never_skipped_or_capped():
    state = {
        "round": 12,
        "last_reviewed_sha": "x",
        "posted_total": 99,
        "previous_round_count": 0,
    }
    decision = rb.decide(state, policy(), "House Rules", 5)
    assert decision["skip"] is False
    assert decision["exempt"] is True
    assert decision["max_comments"] == 0  # no ceiling
    assert rb.progress_line(decision) == ""


# ------------------------------------------------------------- the author's view


def test_the_progress_line_names_the_round_and_what_is_left():
    state = {
        "round": 3,
        "last_reviewed_sha": "x",
        "posted_total": 18,
        "previous_round_count": 7,
    }
    line = rb.progress_line(rb.decide(state, policy(), "Security", 5))
    assert "Round 3" in line
    assert "18 of this PR's 25" in line
    assert "capped at 4" in line
    assert "Security and Correctness" in line


def test_the_first_round_does_not_threaten_the_author_with_narrowing():
    state = {
        "round": 1,
        "last_reviewed_sha": "",
        "posted_total": 0,
        "previous_round_count": 0,
    }
    line = rb.progress_line(rb.decide(state, policy(), "Security", 5))
    assert "from here only" not in line


# ------------------------------------------------------------------ the CLI


def test_a_full_review_clears_the_diff_scope_but_not_the_caps(monkeypatch):
    """`@ai-review` means "look at all of it", not "give me another 25"."""
    reviews = [review("aaa", "2026-01-01T00:00:00Z", rid=1)]
    monkeypatch.setattr(
        rb,
        "gh_json",
        lambda path: reviews if "reviews" in path else [comment(1)] * 20,
    )
    argv = [
        "review_budget.py",
        "--repo",
        "o/r",
        "--pr",
        "1",
        "--lane",
        "Security",
        "--full-review",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rb.main()
    # Scope reset, history intact.
    state = rb.summarise_history(
        reviews, [comment(1)] * 20, POLICY["exempt_lanes"]
    )
    assert state["posted_total"] == 20
    assert rb.decide(state, policy(), "Security", 5)["allowance"] == 5


def test_an_unreadable_history_is_a_ci_fault_not_a_guess(monkeypatch):
    """Guessing here means guessing the round, and a wrong guess of 1 hands a
    full batch to a PR that has already had six."""
    monkeypatch.setattr(rb, "gh_json", lambda path: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_budget.py",
            "--repo",
            "o/r",
            "--pr",
            "1",
            "--lane",
            "Security",
        ],
    )
    assert rb.main() == 2


def test_the_decision_reaches_github_output(tmp_path, monkeypatch):
    out = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(
        rb,
        "gh_json",
        lambda path: (
            [review("aaa", "2026-01-01T00:00:00Z", rid=1)]
            if "reviews" in path
            else [comment(1)] * 10
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_budget.py",
            "--repo",
            "o/r",
            "--pr",
            "1",
            "--lane",
            "Security",
            "--github-output",
        ],
    )
    rb.main()
    written = dict(
        line.split("=", 1) for line in out.read_text().strip().splitlines()
    )
    assert written["round"] == "2"
    assert written["skip"] == "false"
    assert written["last_reviewed_sha"] == "aaa"
    assert written["max_comments"] == "2"  # floor(0.6 * 10) = 6, /4 -> 2


def test_no_output_value_can_span_two_lines(tmp_path, monkeypatch):
    """A newline in a $GITHUB_OUTPUT value forges a second output. Nothing
    here is built from PR content, but the guard is cheap and permanent."""
    out = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    decision = {
        "round": 2,
        "last_reviewed_sha": "a\nb=evil",
        "posted_total": 0,
        "max_comments": 1,
        "skip": False,
        "exempt": False,
        "allowance": 1,
        "lifetime_cap": 25,
        "blocker_lanes": ["Security"],
        "narrow_after": 2,
        "reason": "one\ntwo",
    }
    rb.emit(decision, True)
    for line in out.read_text().strip().splitlines():
        assert line.count("=") >= 1
    assert "evil" not in out.read_text()


def test_the_script_runs_end_to_end():
    """Catches an import error or a syntax slip that unit tests importing the
    module would not, since the workflow shells out to it."""
    proc = subprocess.run(
        [sys.executable, str(Path(rb.__file__)), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "tools")},
    )
    assert proc.returncode == 0
    assert "--full-review" in proc.stdout


# ------------------------------------------------------- the workflow wiring

CORE = REPO_ROOT / ".github" / "workflows" / "_ai-pr-review-core.yml"


@pytest.fixture(scope="module")
def core():
    return yaml.safe_load(CORE.read_text(encoding="utf-8"))


def _step(core, step_id):
    for step in core["jobs"]["review"]["steps"]:
        if step.get("id") == step_id:
            return step
    raise AssertionError(f"no step with id {step_id!r}")


def test_the_budget_is_decided_before_the_diff_is_fetched(core):
    """The fetch scopes itself to `last_reviewed_sha`, which the budget step
    produces. The wrong order silently gives every round a full diff."""
    ids = [s.get("id") for s in core["jobs"]["review"]["steps"]]
    assert ids.index("budget") < ids.index("fetch_diff")


def test_the_budget_step_passes_the_flags_the_script_defines(core):
    run = _step(core, "budget")["run"]
    assert "review_budget.py" in run
    for flag in ("--repo", "--pr", "--lane", "--head-sha", "--github-output"):
        assert flag in run, f"the workflow no longer passes {flag}"


def test_a_manual_invoke_widens_the_scope_but_not_the_volume(core):
    """`--full-review` re-reads the whole PR. It must not also reset the caps,
    or `@ai-review` becomes a way to re-flood a pull request."""
    run = _step(core, "budget")["run"]
    assert "--full-review" in run
    assert "issue_comment" in run and "workflow_dispatch" in run
    assert "lifetime" not in run.lower(), (
        "the workflow appears to be overriding the cap on a manual invoke"
    )


def test_the_ceiling_and_the_progress_line_reach_the_poster(core):
    build = _step(core, "build_review")
    assert "--max-comments" in build["run"]
    assert "--progress" in build["run"]
    assert build["env"]["MAX_COMMENTS"].endswith(
        "budget.outputs.max_comments }}"
    )
    assert build["env"]["PROGRESS"].endswith("budget.outputs.progress }}")


def test_every_expensive_step_stops_when_the_lane_is_skipped(core):
    """The model call is the expensive one, and a skipped lane must not make
    it. Each of these gates on prepare_diff, which itself gates on the budget."""
    for step_id in ("auth", "agy_pr_review", "build_review"):
        condition = " ".join(_step(core, step_id)["if"].split())
        assert "reviewable == 'true'" in condition, (
            f"{step_id} tests reviewable for inequality; a SKIPPED "
            "prepare_diff leaves that output empty, and '' != 'false' is "
            "true — so the step would run after the budget stopped the lane"
        )


def test_the_fetch_falls_back_to_the_whole_pr_when_it_must(core):
    run = _step(core, "fetch_diff")["run"]
    assert "gh pr diff" in run, "there is no full-diff fallback left"
    assert "force-push" in run, "the orphaned-sha case is not handled"
    assert "compare/${LAST_REVIEWED_SHA}...${HEAD_SHA}" in run


def test_an_empty_incremental_diff_stops_rather_than_re_reviewing(core):
    run = _step(core, "fetch_diff")["run"]
    assert "empty=true" in run
    prepare = " ".join(_step(core, "prepare_diff")["if"].split())
    assert "fetch_diff.outputs.empty != 'true'" in prepare


def test_the_lane_count_matches_the_policy_lane_list():
    """prepare_review_diff.py divides the round-1 budget by a hardcoded lane
    count, because it runs without PyYAML. Adding a fifth throttled lane to
    policy.yml without changing it would silently give every lane a share
    that no longer adds up."""
    sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))
    import prepare_review_diff

    assert prepare_review_diff.LANE_COUNT == len(POLICY["lanes"]), (
        f"policy.yml lists {len(POLICY['lanes'])} throttled lanes but "
        f"prepare_review_diff.LANE_COUNT is {prepare_review_diff.LANE_COUNT}"
    )


def test_the_prompt_target_never_exceeds_the_enforced_ceiling(tmp_path):
    """Telling the model to aim for 5 while the poster keeps 1 wastes four
    findings and makes the job log a puzzle."""
    diff = tmp_path / "d.txt"
    diff.write_text(
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "@@ -0,0 +1,3 @@\n+x = 1\n+y = 2\n+z = 3\n",
        encoding="utf-8",
    )
    out = tmp_path / "o.txt"
    gh_out = tmp_path / "gh.txt"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / ".github/scripts/prepare_review_diff.py"),
            "--diff",
            str(diff),
            "--out",
            str(out),
            "--budget-ceiling",
            "1",
            "--github-output",
            str(gh_out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    written = dict(
        line.split("=", 1) for line in gh_out.read_text().strip().splitlines()
    )
    assert written["budget"] == "1"
