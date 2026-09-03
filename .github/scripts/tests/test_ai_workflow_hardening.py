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
"""Pin the security properties of the two agy-driven workflows.

These workflows run a model over input an anonymous contributor wrote — a
fork PR's diff, an issue body — on a runner that holds a Google Cloud
credential and, in one job, a token that can write to the repository. What
keeps that safe is a handful of small facts spread across two YAML files:
an empty tool allowlist, a job that cannot write, a job that cannot reach
the cloud.

Every one of those facts is a line someone can plausibly "fix" later for a
good-sounding local reason — a review that failed on a denied tool, a job
that would be simpler merged back into one. None of them is covered by
ruff, by actionlint, or by any test of the scripts themselves, so without
this file the entire remediation for b/555419958 can be undone silently and
every check stays green.

The assertions are deliberately about PROPERTIES, not about exact text, so
ordinary edits to these workflows do not trip them.
"""

import json
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"

PR_REVIEW = WORKFLOWS / "_ai-pr-review-core.yml"
ISSUE_TRIAGE = WORKFLOWS / "_ai-issue-triage-core.yml"

# Both workflows that hand an untrusted string to an agy agent.
AGENT_WORKFLOWS = [PR_REVIEW, ISSUE_TRIAGE]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _steps(path: Path, job: str) -> list[dict]:
    return _load(path)["jobs"][job]["steps"]


def _code(script: str) -> str:
    """A `run:` block with its comment lines removed.

    These workflows are heavily commented and the comments discuss the very
    things these tests search for — "the agy call", "cat agy_result.json".
    Matching against raw text finds the prose and reports a command that is
    not there, so every search below runs on code only.
    """
    return "\n".join(
        line
        for line in script.splitlines()
        if not line.lstrip().startswith("#")
    )


def _run_blocks(path: Path, job: str | None = None) -> str:
    """Every `run:` script in the workflow (or one job), comments stripped."""
    doc = _load(path)
    jobs = [doc["jobs"][job]] if job else list(doc["jobs"].values())
    return "\n".join(
        _code(step["run"])
        for j in jobs
        for step in j["steps"]
        if isinstance(step.get("run"), str)
    )


def _agy_settings(path: Path) -> dict:
    """The settings.json the workflow writes for agy, parsed.

    Pulled out of the heredoc and JSON-parsed rather than matched as text, so
    reformatting the block does not break the test but changing what it GRANTS
    does. Found by searching every job rather than by naming one, so renaming
    or moving the job does not quietly stop this from checking anything.
    """
    script = next(
        step["run"]
        for job in _load(path)["jobs"].values()
        for step in job["steps"]
        if "antigravity-cli/settings.json" in str(step.get("run", ""))
    )
    body = script.split("<<'SETTINGS_EOF'", 1)[1].split("SETTINGS_EOF", 1)[0]
    return json.loads(body)


# --------------------------------------------------------------------------
# The tool grant — the actual boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", AGENT_WORKFLOWS, ids=lambda p: p.name)
def test_the_agy_tool_allowlist_is_empty(path):
    """The one that matters.

    b/555419958 was reachable because this list held `read_file(*)`,
    `write_file(*)` and `command(cat)` for a step whose own comment said it
    needed no tool. $GOOGLE_APPLICATION_CREDENTIALS is an absolute path
    outside the workspace and the file at it is on its own enough to
    impersonate the service account, so ANY read primitive here is a
    credential disclosure waiting for someone to write the right diff.

    If a review starts failing with "produced no response", fix the prompt.
    Do not make this list non-empty.
    """
    assert _agy_settings(path)["permissions"]["allow"] == []


@pytest.mark.parametrize("path", AGENT_WORKFLOWS, ids=lambda p: p.name)
def test_the_agent_result_is_never_echoed_whole_to_the_log(path):
    """A public repository's workflow log is world-readable, and live.

    Printing the raw response puts model text — derived from attacker-written
    input — on a public channel that GitHub's secret masking does not cover,
    because the credential on this runner is not a registered secret.
    """
    assert "cat agy_result.json" not in _run_blocks(path)


# --------------------------------------------------------------------------
# The job split — no runner holds both halves
# --------------------------------------------------------------------------

WRITE_SCOPES = {"write", "admin"}


def _writes(permissions) -> set[str]:
    if not isinstance(permissions, dict):
        # `write-all`, or inherited — either way, not what we want here.
        return {str(permissions)}
    return {k for k, v in permissions.items() if v in WRITE_SCOPES}


def test_the_job_that_runs_the_model_cannot_write():
    """`review` is the job an injected diff could plausibly reach.

    `id-token: write` is the exception and is not a repository write: it
    mints the OIDC token for Workload Identity Federation. Everything that
    mutates the pull request belongs to `post`.
    """
    assert _writes(_load(PR_REVIEW)["jobs"]["review"]["permissions"]) == {
        "id-token"
    }


def test_the_job_that_can_write_holds_no_cloud_credential():
    """`post` must not authenticate to Google Cloud or run a model.

    Checked structurally — no `id-token`, no auth action, no agy — rather
    than by name, so merging the model call back into this job trips it
    however it is spelled.
    """
    post = _load(PR_REVIEW)["jobs"]["post"]
    assert "id-token" not in (post["permissions"] or {})

    assert not [
        s
        for s in post["steps"]
        if "google-github-actions/auth" in s.get("uses", "")
    ]
    assert "agy " not in _run_blocks(PR_REVIEW, "post")


def test_repo_code_run_after_the_agent_is_integrity_checked():
    """The one step that still executes the checkout in the agent's own job.

    Splitting the posting job off removed the report's "write to a file the
    pipeline executes seconds later" vector from `post`, but `build_review`
    runs `.github/scripts/post_review_comments.py` in the SAME job as the
    agent and after it. With an empty tool allowlist nothing can write there;
    this pins the check that catches it if something can.

    `--porcelain` and not `git diff`: python puts a script's directory on
    sys.path, so a NEW untracked `.github/scripts/json.py` shadows the stdlib
    without modifying any tracked file, and `git diff` would not see it.
    """
    # Comments stripped first: the block above the gate explains itself by
    # naming post_review_comments.py, and partitioning on the raw text would
    # split at the prose rather than at the invocation.
    script = _code(
        next(
            s["run"]
            for s in _steps(PR_REVIEW, "review")
            if s.get("id") == "build_review"
        )
    )
    gate, _, invocation = script.partition("post_review_comments.py")

    assert "git status --porcelain -- .github/scripts" in gate, (
        "the integrity check must run BEFORE the script it protects"
    )
    assert "exit 1" in gate
    assert invocation, "build_review no longer invokes the script"


def test_the_job_that_can_write_takes_no_checkout():
    """So there is no script on its disk for a written file to become.

    The report's code-execution half turned on the agent writing to
    `.github/scripts/post_review_comments.py`, which the next step executed.
    A posting job with no checkout cannot be attacked that way whatever agy's
    `write_file` sandbox turns out to do.
    """
    assert not [
        s
        for s in _steps(PR_REVIEW, "post")
        if "actions/checkout" in s.get("uses", "")
    ]


def test_the_model_and_the_posting_step_are_in_different_jobs():
    """The property the two tests above are each half of."""
    jobs = _load(PR_REVIEW)["jobs"]
    runs_agy = {n for n in jobs if "agy -p" in _run_blocks(PR_REVIEW, n)}
    posts = {
        n
        for n in jobs
        if "--method POST" in _run_blocks(PR_REVIEW, n)
        and "/reviews" in _run_blocks(PR_REVIEW, n)
    }

    # Both non-empty first: an assertion that the two sets are disjoint is
    # trivially satisfied if a rename made one of them empty, which would
    # turn this test green at exactly the moment it stopped checking.
    assert runs_agy, "no job runs agy — has the invocation been renamed?"
    assert posts, "no job posts a review — has the endpoint changed?"
    assert runs_agy.isdisjoint(posts)


# --------------------------------------------------------------------------
# The credential scan
# --------------------------------------------------------------------------


def test_the_response_is_scanned_against_real_credential_material():
    """Not a keyword list, and not the whole ADC file either.

    A keyword list is one rewording away from useless. The whole ADC file is
    the opposite failure: `.type` is "external_account" and `.token_url` is
    the public STS endpoint, and this repository is full of Google Cloud
    recipes whose diffs legitimately contain both — which the reviewer then
    quotes verbatim into `window`. Scanning for those would fail honest PRs
    and cry security incident. Only the secret-bearing fields belong here.
    """
    script = next(
        s["run"]
        for s in _steps(PR_REVIEW, "review")
        if isinstance(s.get("run"), str) and "NEEDLES=" in s["run"]
    )
    assert "ACTIONS_ID_TOKEN_REQUEST_TOKEN" in script
    assert ".credential_source" in script

    # The config fields that must NOT become needles.
    jq_filter = script.split("jq -r '[", 1)[1].split("]", 1)[0]
    for public in (".type", ".token_url", ".audience", ".quota_project_id"):
        assert public not in jq_filter, f"{public} is not secret"


def test_a_response_carrying_the_credential_fails_the_job():
    """The scan must stop the run, not filter and carry on.

    If the response contains this runner's credential, the tool allowlist did
    not hold, and nothing downstream is trustworthy enough to sort the good
    findings from the bad.
    """
    script = next(
        s["run"]
        for s in _steps(PR_REVIEW, "review")
        if isinstance(s.get("run"), str) and "grep -qFf" in s["run"]
    )
    scan = script.split("grep -qFf", 1)[1]
    assert "exit 1" in scan


# --------------------------------------------------------------------------
# The prompt's untrusted-input guard
#
# Hygiene, not a boundary — but the gap it closes is the one the report led
# with, and a silent regression would put the file back where it started.
# --------------------------------------------------------------------------


def test_the_diff_is_marked_untrusted_in_the_prompt():
    script = next(
        s["run"]
        for s in _steps(PR_REVIEW, "review")
        if isinstance(s.get("run"), str) and "## PR Context" in s["run"]
    )
    assert "## Untrusted input" in script

    diff_header = next(
        line for line in script.splitlines() if "Complete unified diff" in line
    )
    assert "UNTRUSTED" in diff_header
