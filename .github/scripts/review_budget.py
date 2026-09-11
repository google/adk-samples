#!/usr/bin/env python3
"""Work out how much one AI review lane may say on this push.

Used by .github/workflows/_ai-pr-review-core.yml, once per lane per run.

WHY THIS EXISTS. Every push re-reviewed the whole pull request, and duplicate
suppression drops anything already said — so each round was FORCED to return
findings nobody had seen yet. On a large PR the pool of available findings is
far bigger than one round's budget, so the reviewer produced new comments
indefinitely. Authors reported the obvious consequence: fix everything, push,
receive a fresh batch, with no sign of an end.

Four things make it converge, and this script decides all four:

  SCOPE      a round reads only what changed since the last review, so a push
             that just fixes comments has almost nothing new to look at
  DECAY      each round's allowance is a fraction of what the last round
             actually posted
  CAP        a hard ceiling on comments per pull request, ever
  NARROWING  past a certain round, only the blocker lanes run at all

Every constant lives in .github/policy.yml under `pr_review_budget`.

NO STORED STATE. Everything is derived from the pull request itself on each
run: the reviews API records the `commit_id` each review was made against, and
each inline comment names the review it belongs to. So "which round is this",
"what did we review last time" and "how many comments have we posted" are all
answerable from two API calls, and nothing has to be persisted between runs or
cleaned up afterwards.

Usage:
  review_budget.py --repo owner/name --pr 123 --lane Correctness \
      [--head-sha SHA] [--full-review] [--churn-budget 5] [--github-output]

Exit codes:
  0  decision written
  2  CI fault — the pull request's review history could not be read
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ci_message import (
    EXIT_OK,
    guard,
    infra_fault,
    report_infra_fault,
)

CHECKER = "review_budget.py"
POLICY_PATH = Path(__file__).resolve().parents[1] / "policy.yml"

# Written into every review body we post, so a later run can recognise its own
# work. The body text is the fallback for reviews posted before this existed,
# but it is prose and prose gets edited; the marker is not meant to be read by
# a human and so will not be.
REVIEW_MARKER = "<!-- adk-ai-review -->"

# How our reviews identified themselves before REVIEW_MARKER. Kept so the round
# counter does not restart at 1 on every pull request that was already under
# review when this shipped. `build_payload` writes this header.
LEGACY_HEADER = re.compile(r"^Automated \*\*[^*]+\*\* review")

# Fallbacks if policy.yml cannot be read. Deliberately conservative: if the
# limits are unreadable we want LESS review, not the unbounded behaviour this
# script exists to end.
DEFAULTS = {
    "lifetime_cap": 25,
    "decay": 0.6,
    "min_allowance": 1,
    "blocker_only_after_round": 2,
    "lanes": ["Security", "Correctness", "Maintainability", "Hygiene"],
    "blocker_lanes": ["Security", "Correctness"],
    "exempt_lanes": ["House Rules"],
}


def load_policy() -> dict:
    """The `pr_review_budget` section, merged over DEFAULTS."""
    try:
        import yaml

        with open(POLICY_PATH, "rb") as handle:
            section = (yaml.safe_load(handle) or {}).get("pr_review_budget")
    except Exception as exc:  # any failure at all means "use the defaults"
        print(f"  could not read {POLICY_PATH}: {exc}; using defaults")
        return dict(DEFAULTS)
    if not isinstance(section, dict):
        print(
            f"  {POLICY_PATH} has no pr_review_budget section; using defaults"
        )
        return dict(DEFAULTS)
    return {**DEFAULTS, **section}


def gh_json(path: str):
    """A paginated GitHub API call, decoded. None if it could not be made."""
    try:
        proc = subprocess.run(
            ["gh", "api", "--paginate", path],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  could not read {path}: {exc}")
        return None
    if proc.returncode != 0:
        print(f"  could not read {path}: {proc.stderr.strip()[:200]}")
        return None
    # --paginate concatenates one JSON array per page, so decode in sequence
    # rather than parsing the output as a single document.
    decoder = json.JSONDecoder()
    items, text, index = [], proc.stdout, 0
    while (start := text.find("[", index)) != -1:
        try:
            batch, index = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            break
        items.extend(batch)
    return items


def is_ours(review: dict) -> bool:
    body = str(review.get("body") or "")
    return REVIEW_MARKER in body or bool(LEGACY_HEADER.match(body.lstrip()))


def lane_of(review: dict) -> str:
    """The lane that posted this review, from its body header."""
    match = re.search(
        r"Automated \*\*([^*]+)\*\* review", str(review.get("body") or "")
    )
    return match.group(1).strip() if match else ""


def summarise_history(reviews: list, comments: list, exempt: list) -> dict:
    """Round number, last reviewed commit, and how much has been said.

    A "round" is a COMMIT we reviewed, not a review we posted: four lanes post
    four reviews against one push, and counting those as four rounds would run
    the decay four times per push and silence the reviewer on the second one.
    """
    ours = [
        r
        for r in reviews
        if is_ours(r) and lane_of(r) not in exempt and r.get("commit_id")
    ]
    ours.sort(key=lambda r: str(r.get("submitted_at") or ""))

    rounds: list[str] = []
    for review in ours:
        if review["commit_id"] not in rounds:
            rounds.append(review["commit_id"])

    review_commit = {r.get("id"): r.get("commit_id") for r in ours}
    our_review_ids = set(review_commit)

    posted_total = 0
    previous_round_count = 0
    last_sha = rounds[-1] if rounds else ""
    for comment in comments:
        review_id = comment.get("pull_request_review_id")
        if review_id not in our_review_ids:
            continue
        posted_total += 1
        if review_commit.get(review_id) == last_sha:
            previous_round_count += 1

    return {
        "round": len(rounds) + 1,
        "last_reviewed_sha": last_sha,
        "posted_total": posted_total,
        "previous_round_count": previous_round_count,
    }


def allocate(allowance: int, lanes: list, lane: str) -> int:
    """This lane's share, dealt one comment at a time down the priority list.

    Not `allowance // len(lanes)`: the lanes are concurrent jobs that cannot
    coordinate, so each computes the whole vector and reads its own slot. An
    allowance of 1 must go to ONE lane. Dividing and letting each lane round up
    to at least one turns an allowance of 1 into four comments, which is the
    tail of the decay curve — exactly where being exact matters most.
    """
    if lane not in lanes:
        return allowance
    index = lanes.index(lane)
    base, extra = divmod(max(0, allowance), len(lanes))
    return base + (1 if index < extra else 0)


def decide(
    state: dict,
    policy: dict,
    lane: str,
    churn_budget: int,
    head_sha: str = "",
) -> dict:
    """The whole decision for this lane: how many comments, and whether to run."""
    lanes = list(policy["lanes"])
    exempt = list(policy["exempt_lanes"])
    blockers = list(policy["blocker_lanes"])
    cap = int(policy["lifetime_cap"])
    decay = float(policy["decay"])
    floor = int(policy["min_allowance"])
    narrow_after = int(policy["blocker_only_after_round"])

    # Nothing has moved since the last round. A re-run, a label change, a
    # comment: none of them is new code, and re-reviewing the same commit is
    # how the reviewer used to produce a second batch of comments about work
    # the author had not touched.
    if head_sha and head_sha == state["last_reviewed_sha"]:
        return {
            **state,
            "lane": lane,
            "skip": True,
            "max_comments": 0,
            "exempt": lane in exempt,
            "allowance": 0,
            "remaining": 0,
            "lifetime_cap": cap,
            "blocker_lanes": blockers,
            "narrow_after": narrow_after,
            "reason": f"commit {head_sha[:8]} has already been reviewed",
        }

    if lane in exempt:
        return {
            **state,
            "lane": lane,
            "skip": False,
            "max_comments": 0,  # 0 means "no ceiling" for an exempt lane
            "exempt": True,
            "reason": "exempt from the review budget",
        }

    round_number = state["round"]
    remaining = max(0, cap - state["posted_total"])

    if round_number <= 1:
        allowance = churn_budget * len(lanes)
        basis = f"first round, {churn_budget} per lane"
    else:
        decayed = math.floor(decay * state["previous_round_count"])
        allowance = max(floor, decayed)
        basis = (
            f"{state['previous_round_count']} comment(s) last round x {decay}"
        )

    allowance = min(allowance, remaining)
    mine = allocate(allowance, lanes, lane)

    skip = False
    reason = f"round {round_number}: {basis}"
    if remaining <= 0:
        skip = True
        reason = (
            f"this PR has had {state['posted_total']} automated comments, "
            f"at the {cap} limit"
        )
    elif round_number > narrow_after and lane not in blockers:
        skip = True
        reason = (
            f"round {round_number}: past round {narrow_after}, only "
            f"{' and '.join(blockers)} still run"
        )
    elif mine <= 0:
        skip = True
        reason = (
            f"round {round_number}: the {allowance}-comment allowance went to "
            f"higher-priority lanes"
        )

    return {
        **state,
        "lane": lane,
        "skip": skip,
        "max_comments": mine,
        "exempt": False,
        "allowance": allowance,
        "remaining": remaining,
        "lifetime_cap": cap,
        "blocker_lanes": blockers,
        "narrow_after": narrow_after,
        "reason": reason,
    }


def progress_line(decision: dict) -> str:
    """The one line an author reads to see that this process ends.

    "Endless" is partly not knowing whether it converges. None of the limits
    above are visible from the outside unless something says so.
    """
    if decision.get("exempt"):
        return ""
    parts = [f"Round {decision['round']}"]
    posted = decision["posted_total"]
    cap = decision["lifetime_cap"]
    parts.append(f"{posted} of this PR's {cap} automated comments used")
    parts.append(f"this round is capped at {decision['allowance']}")
    if decision["round"] >= decision["narrow_after"]:
        parts.append(
            "from here only " + " and ".join(decision["blocker_lanes"]) + " run"
        )
    return " · ".join(parts) + "."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="How much may this review lane say on this push?"
    )
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--lane", required=True, help="e.g. Correctness")
    parser.add_argument(
        "--churn-budget",
        type=int,
        default=5,
        help="per-lane budget from prepare_review_diff.py, used in round 1",
    )
    parser.add_argument(
        "--head-sha",
        default="",
        help="the commit about to be reviewed; if a previous round already "
        "reviewed it, this run has nothing to do",
    )
    parser.add_argument(
        "--full-review",
        action="store_true",
        help="a maintainer asked for this run; read the whole PR again. The "
        "caps still apply — an explicit ask widens the scope, not the volume",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="append the decision to $GITHUB_OUTPUT as well as stdout",
    )
    return parser


def emit(decision: dict, to_github_output: bool) -> None:
    fields = {
        "round": decision["round"],
        "skip": str(decision["skip"]).lower(),
        "max_comments": decision["max_comments"],
        "last_reviewed_sha": decision["last_reviewed_sha"],
        "posted_total": decision["posted_total"],
        "progress": progress_line(decision),
        "reason": decision["reason"],
    }
    print(json.dumps(fields, indent=1))
    if not to_github_output:
        return
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        print("  no $GITHUB_OUTPUT to write to")
        return
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in fields.items():
            # Single-line values only. `progress` and `reason` are built here
            # from numbers and lane names, never from PR content, so neither
            # can carry a newline that would forge a second output.
            handle.write(
                f"{key}={str(value).splitlines()[0] if value else ''}\n"
            )


def main() -> int:
    args = build_parser().parse_args()
    policy = load_policy()

    reviews = gh_json(f"repos/{args.repo}/pulls/{args.pr}/reviews?per_page=100")
    comments = gh_json(
        f"repos/{args.repo}/pulls/{args.pr}/comments?per_page=100"
    )
    if reviews is None or comments is None:
        return report_infra_fault(
            infra_fault(
                CHECKER,
                f"cannot read the review history of {args.repo}#{args.pr}; "
                "refusing to guess a budget",
            )
        )

    state = summarise_history(reviews, comments, policy["exempt_lanes"])
    if args.full_review:
        # An explicit re-review reads everything again, so there is no "last
        # reviewed" point to diff against. The round number and the caps are
        # untouched: asking for another look is not asking for another 25.
        state["last_reviewed_sha"] = ""

    decision = decide(
        state, policy, args.lane, args.churn_budget, args.head_sha
    )
    emit(decision, args.github_output)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(guard(CHECKER, main))
