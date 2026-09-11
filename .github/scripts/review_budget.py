#!/usr/bin/env python3
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
  0  always, including when the history could not be read. There is no failure
     mode here worth a red check: the worst case is a lane that stays quiet for
     one round, and the next push recovers it.
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
    return _validated({**DEFAULTS, **section})


def _validated(policy: dict) -> dict:
    """Every value the right type and in range, or that key falls back.

    Guarding only the READ of policy.yml is not enough. `decide` does
    `float(policy["decay"])` and `list(policy["lanes"])`, so `decay: sixty` or
    `lanes:` left empty raises out of `guard()` as a CI fault -- four red
    checks on every push, from a typo in a config file, in a script whose
    documented contract is that it never fails a build.
    """
    checked = dict(policy)

    def fall_back(key, why):
        print(f"  policy.yml pr_review_budget.{key} {why}; using the default")
        checked[key] = DEFAULTS[key]

    for key in ("lifetime_cap", "min_allowance", "blocker_only_after_round"):
        try:
            checked[key] = int(checked[key])
        except (TypeError, ValueError):
            fall_back(key, "is not a whole number")
            continue
        if checked[key] < 0:
            fall_back(key, "is negative")

    try:
        checked["decay"] = float(checked["decay"])
    except (TypeError, ValueError):
        fall_back("decay", "is not a number")
    if not 0 < checked["decay"] <= 1:
        fall_back("decay", "is outside (0, 1]")

    for key in ("lanes", "blocker_lanes", "exempt_lanes"):
        value = checked.get(key)
        if not isinstance(value, list) or not all(
            isinstance(v, str) for v in value
        ):
            fall_back(key, "is not a list of strings")
    if not checked["lanes"]:
        fall_back("lanes", "is empty")

    # blocker_lanes must be a PREFIX of lanes, or the lanes that survive a
    # shrinking allowance are not the ones that survive the round limit, and
    # the reviewer narrows to one set while allocating to another.
    prefix = checked["lanes"][: len(checked["blocker_lanes"])]
    if checked["blocker_lanes"] and prefix != checked["blocker_lanes"]:
        fall_back("blocker_lanes", "is not a prefix of lanes")
        checked["lanes"] = DEFAULTS["lanes"]
    return checked


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
    if "[" not in text:
        # A 200 whose body is not a JSON array: an HTML rate-limit page, an
        # error object, an empty response from a proxy. Returning [] here
        # reads as "this PR has never been reviewed" and hands a fresh batch
        # to a PR that has already had five rounds -- exactly the failure this
        # file exists to prevent. None means "could not tell", and the caller
        # degrades to silence.
        print(f"  {path} returned no JSON array: {text.strip()[:120]!r}")
        return None
    while (start := text.find("[", index)) != -1:
        try:
            batch, index = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            # A later page that will not decode leaves a PARTIAL history,
            # which under-counts the round and the comments posted. Half an
            # answer is worse than none: it is confidently wrong.
            print(f"  {path} returned a page that could not be decoded")
            return None
        items.extend(batch)
    return items


def is_ours(review: dict) -> bool:
    """Did WE post this review?

    The author check is not decoration. Body text alone is forgeable by anyone
    with read access: a PR author who submits a one-line review containing the
    marker gets it stamped with the current head, and every lane then skips
    with "commit X has already been reviewed". Repeat after each push and the
    reviewer is off for that pull request permanently. Twenty-five forged
    inline comments does the same thing through the lifetime cap.

    Only a GitHub App or Actions token can post as an account of type `Bot`,
    which is exactly the set of things that can be us.
    """
    if str((review.get("user") or {}).get("type") or "") != "Bot":
        return False
    body = str(review.get("body") or "")
    return REVIEW_MARKER in body or bool(LEGACY_HEADER.match(body.lstrip()))


def lane_of(review: dict) -> str:
    """The lane that posted this review, from its body header."""
    # `match`, not `search`, and against the same lstripped body `is_ours`
    # tests: a human quoting one of our reviews should not have their comment
    # attributed to the exempt lane and dropped from the round count.
    body = str(review.get("body") or "").lstrip()
    body = (
        body[len(REVIEW_MARKER) :].lstrip()
        if body.startswith(REVIEW_MARKER)
        else body
    )
    match = re.match(r"Automated \*\*([^*]+)\*\* review", body)
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
    # The commit of the LATEST review, not the last first-seen commit. After a
    # force-push back to an already-reviewed commit A the history is [A, B]
    # but the newest review is against A, and `rounds[-1]` would name B — a
    # commit no longer reachable, whose compare then 404s into a silent full
    # re-review of the whole PR.
    last_sha = ours[-1]["commit_id"] if ours else ""
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
    if not lanes:
        return 0
    if lane not in lanes:
        # A lane label that policy.yml does not list is a misconfiguration --
        # a typo, a renamed workflow, a fifth lane nobody added to the list.
        # Returning the whole allowance let every such lane take all of it, so
        # four mislabelled lanes could post four times the round's budget.
        # Zero is the safe reading: a lane nobody budgeted for has no budget.
        print(
            f"  lane {lane!r} is not in policy.yml's lanes {lanes}; "
            "treating its budget as zero"
        )
        return 0
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

    # Allocate among the lanes that will ACTUALLY RUN this round, not among
    # all four. Past the narrowing round two of them skip, so dividing by four
    # threw away half the allowance -- and the progress line, which reports
    # the undivided number, told the author twice what could be spent.
    narrowed = round_number > narrow_after
    running = (
        [lane_name for lane_name in lanes if lane_name in blockers]
        if narrowed
        else lanes
    )
    mine = allocate(allowance, running, lane) if lane in running else 0

    skip = False
    reason = f"round {round_number}: {basis}"
    if remaining <= 0:
        skip = True
        reason = (
            f"this PR has had {state['posted_total']} automated comments, "
            f"at the {cap} limit"
        )
    elif narrowed and lane not in blockers:
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
        "running_lanes": running,
        "reason": reason,
    }


def progress_line(decision: dict) -> str:
    """The one line an author reads to see that this process ends.

    "Endless" is partly not knowing whether it converges. None of the limits
    above are visible from the outside unless something says so.
    """
    if decision.get("exempt") or decision.get("skip"):
        return ""
    parts = [f"Round {decision['round']}"]
    posted = decision["posted_total"]
    cap = decision["lifetime_cap"]
    parts.append(f"{posted} of this PR's {cap} automated comments used")
    # "across all reviewers", because this same line appears on each lane's
    # review. Without it, four reviews each saying "capped at 20" read as a
    # threat of eighty comments rather than a promise of twenty.
    parts.append(
        f"this round is capped at {decision['allowance']} across all reviewers"
    )
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
            #
            # Truthiness is the wrong test: `max_comments` is legitimately 0,
            # and writing that as an empty string hands the workflow
            # `--max-comments ""`, which argparse rejects as not an integer.
            text = "" if value is None else str(value)
            handle.write(f"{key}={text.splitlines()[0] if text else ''}\n")


def main() -> int:
    args = build_parser().parse_args()
    policy = load_policy()

    reviews = gh_json(f"repos/{args.repo}/pulls/{args.pr}/reviews?per_page=100")
    comments = gh_json(
        f"repos/{args.repo}/pulls/{args.pr}/comments?per_page=100"
    )
    if reviews is None or comments is None:
        # Degrade, never fail — but degrade toward SILENCE, not toward a full
        # batch. Every number here is derived from the review history, so
        # without it the honest options are "assume round 1" and "say
        # nothing". Assuming round 1 hands a fresh 20 comments to a PR that
        # has already had six, which is the exact failure this script exists
        # to prevent; saying nothing costs one round of review and the next
        # push recovers it.
        print(
            f"  cannot read the review history of {args.repo}#{args.pr}; "
            "skipping this lane rather than risking a full batch"
        )
        emit(
            {
                "round": 0,
                "last_reviewed_sha": "",
                "posted_total": 0,
                "lane": args.lane,
                "skip": True,
                "max_comments": 0,
                "exempt": False,
                "reason": "the PR's review history could not be read",
            },
            args.github_output,
        )
        return EXIT_OK

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
